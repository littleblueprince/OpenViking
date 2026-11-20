#!/usr/bin/env python3

import os
import sys
import time
import json
import uuid
import hashlib
import subprocess
import atexit
import urllib.request
import urllib.error
import threading
import asyncio
import contextlib
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Any, Iterable, Optional


sys.path.append(str(Path(__file__).parent))

from core.logger import get_logger
from core.evidence_matcher import EvidenceMatcher
from core.readable_trace import tool_results_from_events, validate_preview_chars
from core.trace_annotation import EvidenceTraceRecorder
from core.trace_models import ChunkDocument
from core.tool_event_bridge import ToolEventEvidenceBridge

logger = get_logger()

_OV_CONF_PATH = str((Path(__file__).parent.parent / "ov.conf").resolve())
_OPENVIKING_SERVER_PROCESS: Optional[subprocess.Popen] = None
_CURRENT_OV_CONF_PATH: Optional[str] = None
_SERVER_LOCK = threading.Lock()
_RUN_LOCK = threading.Lock()


def _generate_temp_ov_conf(original_conf_path: str, vector_store_path: str) -> str:
    """
    Generate a temporary ov.conf file with the specified vector store path.
    
    Args:
        original_conf_path: Path to the original ov.conf file
        vector_store_path: Path to the vector store directory
        
    Returns:
        Path to the temporary ov.conf file
    """
    # Read original config
    with open(original_conf_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # Ensure root_api_key is a string (not null) for Pydantic validation
    if 'server' not in config:
        config['server'] = {}
    if config['server'].get('root_api_key') is None:
        config['server']['root_api_key'] = ""
    
    # Update storage workspace to point to vector store
    if 'storage' not in config:
        config['storage'] = {}
    config['storage']['workspace'] = vector_store_path
    
    # Create temporary config file with stable name based on vector_store_path
    temp_dir = Path(__file__).parent.parent / ".temp"
    temp_dir.mkdir(exist_ok=True)
    
    # 使用 vector_store_path 的哈希值作为稳定的文件名
    # 这样相同的 vector_store 会使用同一个临时配置文件
    vector_store_path_bytes = vector_store_path.encode('utf-8')
    path_hash = hashlib.md5(vector_store_path_bytes).hexdigest()
    temp_conf_path = str(temp_dir / f"ov_{path_hash}.conf")
    
    # 只有当文件不存在或者配置内容变化时才重新写入
    need_write = True
    if os.path.exists(temp_conf_path):
        try:
            with open(temp_conf_path, 'r', encoding='utf-8') as f:
                existing_config = json.load(f)
            if existing_config == config:
                need_write = False
        except Exception:
            need_write = True
    
    if need_write:
        # Write temporary config
        with open(temp_conf_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
    
    return temp_conf_path


def _healthcheck(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=1.5) as resp:
            return 200 <= resp.status < 300
    except Exception:
        return False


def _load_server_url(ov_conf_path: str, *, external: bool = False) -> str:
    with open(ov_conf_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if external:
        url = data.get("bot", {}).get("ov_server", {}).get("server_url")
        if url:
            return os.path.expandvars(url).rstrip("/")
    server = data.get("server", {}) if isinstance(data, dict) else {}
    host = server.get("host", "127.0.0.1")
    port = server.get("port", 1933)
    return f"http://{host}:{port}"


def _stop_openviking_server() -> None:
    global _OPENVIKING_SERVER_PROCESS, _CURRENT_OV_CONF_PATH
    proc = _OPENVIKING_SERVER_PROCESS
    # Only consider "killing all servers" if this runner started one in this process.
    started_by_us = bool(proc) or bool(_CURRENT_OV_CONF_PATH)
    _OPENVIKING_SERVER_PROCESS = None
    _CURRENT_OV_CONF_PATH = None
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()
    
    # Optional safety measure: kill all openviking-server processes.
    #
    # This is intentionally disabled by default to avoid killing a server that was started
    # outside of this benchmark process (e.g. a developer's local OV server).
    # To enable it, set `OPENVIKING_BENCH_KILL_ALL_SERVERS=1`.
    #
    # We also only do this if this process started an OV server at least once.
    kill_all = os.environ.get("OPENVIKING_BENCH_KILL_ALL_SERVERS", "").strip() in ("1", "true", "True")
    if not (kill_all and started_by_us):
        return

    # 额外的安全措施：杀死所有 openviking-server 进程
    try:
        if sys.platform == "darwin" or sys.platform.startswith("linux"):
            # 使用 pgrep 和 pkill 在 macOS 和 Linux 上
            result = subprocess.run(
                ["pgrep", "-f", "openviking-server"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                logger.info(f"Found openviking-server processes: {result.stdout.strip()}")
                subprocess.run(["pkill", "-f", "openviking-server"], capture_output=True)
                time.sleep(1)
    except Exception as e:
        logger.debug(f"Failed to kill all openviking-server processes: {e}")


atexit.register(_stop_openviking_server)


def _ensure_openviking_server(ov_conf_path: str) -> None:
    global _OPENVIKING_SERVER_PROCESS, _CURRENT_OV_CONF_PATH

    with _SERVER_LOCK:
        server_url = _load_server_url(ov_conf_path)
        health_url = f"{server_url}/health"

        # 检查是否已经有正确配置的服务器在运行
        if (_CURRENT_OV_CONF_PATH == ov_conf_path and 
            _OPENVIKING_SERVER_PROCESS and 
            _OPENVIKING_SERVER_PROCESS.poll() is None and 
            _healthcheck(health_url)):
            # 已有正确配置的服务器在运行，直接返回
            return

        # 需要启动新服务器，先停止旧的
        _stop_openviking_server()
        _CURRENT_OV_CONF_PATH = None

        if _healthcheck(health_url):
            raise RuntimeError("An OpenViking server already uses this address. Set manage_server=false to reuse it.")

        # 确保旧进程真的被终止
        time.sleep(0.5)

        # 启动新服务器
        env = os.environ.copy()
        env["OPENVIKING_CONFIG_FILE"] = ov_conf_path
        _OPENVIKING_SERVER_PROCESS = subprocess.Popen(
            ["openviking-server", "--config", ov_conf_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
        )
        _CURRENT_OV_CONF_PATH = ov_conf_path

        # 等待服务器健康
        deadline = time.time() + 20
        while time.time() < deadline:
            if _OPENVIKING_SERVER_PROCESS.poll() is not None:
                raise RuntimeError("openviking-server exited unexpectedly")
            if _healthcheck(health_url):
                return
            time.sleep(0.3)

        raise RuntimeError("openviking-server did not become healthy in time")


def _build_vikingbot_env(ov_conf_path: str, base_conf_path: str) -> dict[str, str]:
    env = {"OPENVIKING_CONFIG_FILE": ov_conf_path}
    
    # 设置 ovcli.conf 的路径，和原始 ov.conf 在同一个目录（不是临时文件的目录）
    ovcli_conf_path = Path(base_conf_path).with_name("ovcli.conf")
    if ovcli_conf_path.is_file():
        env["OPENVIKING_CLI_CONFIG_FILE"] = str(ovcli_conf_path)
    
    # Read API key from ov.conf
    try:
        with open(ov_conf_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        if 'vlm' in config and 'api_key' in config['vlm']:
            api_key = config['vlm']['api_key']
            env['OPENAI_API_KEY'] = os.path.expandvars(api_key)
    except Exception as e:
        logger.warning(f"Failed to read API key from ov.conf: {e}")
    
    return env


@contextlib.contextmanager
def _temporary_environ(updates: dict[str, str]):
    old_values: dict[str, Optional[str]] = {}
    try:
        for key, value in updates.items():
            old_values[key] = os.environ.get(key)
            os.environ[key] = value
        yield
    finally:
        for key, old in old_values.items():
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old


def _run_vikingbot_in_process(
    input_msg: str,
    session_id: str,
    ov_conf_path: str,
    base_conf_path: str,
    max_iterations: int,
    disabled_tools: list[str],
    timeout: float,
    events: list[dict[str, Any]],
    event_handler: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Submit one native agent turn and capture its outbound events."""
    repo_root = Path(__file__).resolve().parents[3]
    bot_root = repo_root / "bot"
    if str(bot_root) not in sys.path:
        sys.path.insert(0, str(bot_root))

    from openviking.utils.async_client_cache import LoopScopedAsyncClientCache
    from vikingbot.bus.events import InboundMessage
    from vikingbot.bus.queue import MessageBus
    from vikingbot.channels.single_turn import SingleTurnChannelConfig
    from vikingbot.cli.commands import _init_bot_data, prepare_agent_loop, validate_openviking_auth
    from vikingbot.config import loader
    from vikingbot.config.schema import SessionKey
    from vikingbot.session.manager import SessionManager
    from vikingbot.utils import helpers

    async def run_once(config) -> dict[str, Any]:
        bus = MessageBus()
        agent_loop = prepare_agent_loop(
            config=config,
            bus=bus,
            session_manager=SessionManager(config.bot_data_path),
            cron=None,
            quiet=True,
            eval=True,
        )
        session_key = SessionKey(
            type="cli", channel_id=SingleTurnChannelConfig().channel_id(), chat_id=session_id
        )
        started = time.monotonic()
        await bus.publish_inbound(
            InboundMessage(
                sender_id="user", content=input_msg, session_key=session_key,
                metadata={"disabled_tools": disabled_tools},
            )
        )

        async def collect_response() -> dict[str, Any]:
            while True:
                msg = await bus.consume_outbound()
                if msg.session_key != session_key:
                    continue
                kind = msg.event_type.value
                elapsed = round(time.monotonic() - started, 3)
                value: Any = msg.content
                if kind == "response":
                    value = {
                        "content": msg.content, "response_id": msg.response_id,
                        "token_usage": msg.token_usage, "time_cost": msg.time_cost,
                        "iteration": msg.iteration, "tools_used_names": msg.tools_used_names,
                    }
                elif kind == "interrupt":
                    value = msg.metadata
                elif kind not in {
                    "iteration", "tool_call", "tool_result", "content_delta", "error"
                }:
                    continue
                event = {
                    "event": kind, "elapsed_seconds": elapsed,
                    "data": {
                        "timestamp": datetime.now().isoformat(), "value": value,
                        "metadata": msg.metadata, "response_id": msg.response_id,
                        "iteration": msg.iteration, "session_id": session_key.safe_name(),
                    },
                }
                events.append(event)
                if event_handler is not None:
                    event_handler(event)
                if kind == "response":
                    return {**value, "elapsed_seconds": elapsed}
                if kind in {"interrupt", "error"}:
                    raise RuntimeError(f"VikingBot turn ended with {kind}")

        agent_task = asyncio.create_task(agent_loop.run())
        response_task = asyncio.create_task(collect_response())
        try:
            done, _ = await asyncio.wait(
                {agent_task, response_task}, timeout=timeout, return_when=asyncio.FIRST_COMPLETED,
            )
            if response_task in done:
                return response_task.result()
            if agent_task in done:
                agent_task.result()
                raise RuntimeError("VikingBot stopped before returning an answer")
            raise TimeoutError(f"VikingBot did not answer within {timeout:g} seconds")
        finally:
            for task in (agent_task, response_task):
                task.cancel()
            await asyncio.gather(agent_task, response_task, return_exceptions=True)
            try:
                await agent_loop.close_mcp()
            finally:
                await LoopScopedAsyncClientCache.close_current_loop_clients()

    env = _build_vikingbot_env(ov_conf_path, base_conf_path)
    old_config_path, old_bot_path = loader.CONFIG_PATH, helpers._bot_data_path
    with _temporary_environ(env):
        try:
            config = loader.ensure_config(Path(ov_conf_path))
            validate_openviking_auth(config)
            _init_bot_data(config)
            config.agents.max_tool_iterations = max_iterations
            return asyncio.run(run_once(config))
        finally:
            loader.CONFIG_PATH = old_config_path
            helpers._bot_data_path = old_bot_path


class VikingBotRunner:
    """
    Wrapper for VikingBot to support Agentic RAG evaluation.
    
    This class provides a simple interface to run VikingBot
    for generating answers using Agentic RAG approach.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize VikingBotRunner.
        
        Args:
            config: Configuration dictionary containing vikingbot settings
        """
        self.config = config
        self.vikingbot_config = config.get('vikingbot', {})
        self.max_iterations = int(self.vikingbot_config.get('max_iterations', 50))
        self.timeout = float(self.vikingbot_config.get('timeout_seconds', 330))
        self.evidence_preview_chars = self.vikingbot_config.get('evidence_preview_chars', 500)
        validate_preview_chars(self.evidence_preview_chars)
        if self.max_iterations < 1 or self.timeout <= 0:
            raise ValueError("max_iterations and timeout_seconds must be positive")
        self.ov_conf_path = str(
            Path(self.vikingbot_config.get('ov_conf') or _OV_CONF_PATH).expanduser().resolve()
        )
        self.manage_server = self.vikingbot_config.get('manage_server', True)
        # Get vector store path from config if available
        self.vector_store_path = config.get('paths', {}).get('vector_store')
        # Get custom prompt from config if available
        self.custom_prompt = self.vikingbot_config.get('prompt', None)
        self.base_instruction = self.vikingbot_config.get('base_instruction', None)

    def build_prompt(self, question: str, resource_uri: str | None = None) -> str:
        if self.custom_prompt:
            prompt = self.custom_prompt.format(question=question, resource_uri=resource_uri or "")
            if not resource_uri or "{resource_uri}" in self.custom_prompt:
                return prompt
        else:
            instruction = self.base_instruction or (
                "Answer this question as briefly as possible. Use only the information available "
                "in the database. Do not use web search or any external source."
            )
            prompt = f"{instruction}\n\nQuestion: {question}"
        if resource_uri:
            return f"Use only the course syllabus under {resource_uri}.\n\n{prompt}"
        return prompt

    def generate_answer(
        self,
        question: str,
        session_id: Optional[str] = None,
        *,
        resource_uri: str | None = None,
        disabled_tools: list[str] | None = None,
        evidence_matcher: EvidenceMatcher | None = None,
        evidence_trace_path: str | Path | None = None,
        case_id: str | None = None,
        run_id: str | None = None,
    ) -> Dict[str, Any]:
        """Generate an answer through the native agent loop with per-turn tool restrictions."""
        session_id = session_id or f"eval_{uuid.uuid4().hex}"
        started = time.monotonic()
        events: list[dict[str, Any]] = []
        empty_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        recorder = None
        bridge = None
        if evidence_matcher is not None:
            if not evidence_trace_path or not case_id or not run_id:
                raise ValueError("Evidence matching requires trace path, case_id and run_id")
            recorder = EvidenceTraceRecorder(evidence_trace_path, evidence_matcher)
            bridge = ToolEventEvidenceBridge(
                recorder, case_id=case_id, run_id=run_id, logger=logger,
                preview_chars=self.evidence_preview_chars,
            )
        elif evidence_trace_path is not None:
            raise ValueError("Evidence trace path requires a matcher")

        def evidence_output(
            *, answer: str, model_iterations: int | None = None, error: str | None = None
        ) -> dict[str, Any]:
            if recorder is None:
                return {}
            readable_path = recorder.export_readable_json(
                recorder.path.with_name(recorder.path.stem + ".readable.json"),
                preview_chars=self.evidence_preview_chars,
                tool_results=tool_results_from_events(events),
                context={
                    "case_id": case_id, "run_id": run_id, "question": question,
                    "answer": answer, "model_iterations": model_iterations, "error": error,
                    "pending_tool_call_ids": bridge.pending_tool_call_ids,
                },
            )
            return {
                "evidence_trace_path": str(recorder.path),
                "evidence_readable_path": str(readable_path),
                "evidence_summary": recorder.summary().to_dict(),
                "evidence_pending_tool_call_ids": bridge.pending_tool_call_ids,
            }

        try:
            # Native configuration and bot data paths are process-global.
            with _RUN_LOCK:
                ov_conf_path = self.ov_conf_path
                if self.vector_store_path:
                    ov_conf_path = _generate_temp_ov_conf(ov_conf_path, self.vector_store_path)
                if self.manage_server:
                    _ensure_openviking_server(ov_conf_path)
                elif not _healthcheck(f"{_load_server_url(ov_conf_path, external=True)}/health"):
                    raise RuntimeError("The configured OpenViking server is not healthy")
                response = _run_vikingbot_in_process(
                    input_msg=self.build_prompt(question, resource_uri),
                    session_id=session_id,
                    ov_conf_path=ov_conf_path,
                    base_conf_path=self.ov_conf_path,
                    max_iterations=self.max_iterations,
                    disabled_tools=list(disabled_tools or ()),
                    timeout=self.timeout,
                    events=events,
                    event_handler=bridge.consume if bridge else None,
                )
            return {
                "answer": response["content"],
                "total_time_sec": response["time_cost"],
                "elapsed_seconds": response["elapsed_seconds"],
                "response_id": response["response_id"],
                "token_usage": response["token_usage"] or empty_usage,
                "tools_used_names": response["tools_used_names"],
                "iterations_used": response["iteration"],
                "events": events,
                **evidence_output(
                    answer=response["content"], model_iterations=response["iteration"]
                ),
            }
        except Exception as e:
            logger.error(f"Error generating answer with VikingBot: {e}")
            return {
                "answer": f"[ERROR] {str(e)}",
                "error": str(e),
                "total_time_sec": time.monotonic() - started,
                "token_usage": empty_usage,
                "events": events,
                **evidence_output(answer=f"[ERROR] {str(e)}", error=str(e)),
            }
        finally:
            if recorder is not None:
                recorder.close()

    def generate_qa(
        self,
        qa: Any,
        chunks: Iterable[ChunkDocument],
        *,
        case_id: str,
        run_id: str,
        trace_dir: str | Path,
        resource_uri: str | None = None,
        disabled_tools: list[str] | None = None,
        session_id: str | None = None,
    ) -> Dict[str, Any]:
        """Evaluate an adapter QA against caller-supplied full L2 snapshot chunks.

        Snapshot preparation belongs to the benchmark and happens before the
        agent turn. Only the question/resource URI reach the agent.
        """
        matcher = EvidenceMatcher(qa.gold_evidence, chunks, snapshot_id=run_id)
        trace_dir = Path(trace_dir)
        trace_dir.mkdir(parents=True, exist_ok=True)
        artifact_id = uuid.uuid4().hex
        manifest_path = trace_dir / f"{artifact_id}.manifest.json"
        with manifest_path.open("x", encoding="utf-8") as file:
            json.dump(matcher.manifest.to_dict(), file, ensure_ascii=False, indent=2)
        result = self.generate_answer(
            qa.question, session_id, resource_uri=resource_uri, disabled_tools=disabled_tools,
            evidence_matcher=matcher, evidence_trace_path=trace_dir / f"{artifact_id}.steps.jsonl",
            case_id=case_id, run_id=run_id,
        )
        result["evidence_manifest_path"] = str(manifest_path)
        return result


def stop_openviking_server() -> None:
    """
    Stop the currently running OpenViking server.
    This function is exposed for pipeline.py to call.
    """
    _stop_openviking_server()


def run_vikingbot_query(
    question: str,
    config: Dict[str, Any],
    session_id: Optional[str] = None,
    *,
    resource_uri: str | None = None,
    disabled_tools: list[str] | None = None,
    evidence_matcher: EvidenceMatcher | None = None,
    evidence_trace_path: str | Path | None = None,
    case_id: str | None = None,
    run_id: str | None = None,
) -> Dict[str, Any]:
    """
    Synchronous wrapper for VikingBotRunner.generate_answer.
    
    Args:
        question: The question to answer
        config: Configuration dictionary
        session_id: Optional session identifier
        
    Returns:
        Dictionary containing answer and metadata
    """
    runner = VikingBotRunner(config)
    return runner.generate_answer(
        question, session_id, resource_uri=resource_uri, disabled_tools=disabled_tools,
        evidence_matcher=evidence_matcher, evidence_trace_path=evidence_trace_path,
        case_id=case_id, run_id=run_id,
    )
