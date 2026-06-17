import asyncio
import json
import logging
import os
import sys
import tempfile

from autogen_agentchat.agents import AssistantAgent
from autogen_core import CancellationToken
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.tools.mcp import McpWorkbench, StdioServerParams

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class _SuppressLLMStreamStart(logging.Filter):
    def filter(self, record):
        return '"type": "LLMStreamStart"' not in record.getMessage()

logging.getLogger("autogen_core.events").addFilter(_SuppressLLMStreamStart())

OPENAI_API_KEY       = os.environ.get("OPENAI_API_KEY", "")
# MODEL_NAME           = "gpt-4.1"
# MODEL_NAME           = "gpt-5"
MODEL_NAME = "gpt-5-mini"
PLAYWRIGHT_MCP_PATH  = "/Users/gishwin.biju/Desktop/development/MCP/playwright-mcp/packages/playwright-mcp/cli.js"

LOGIN_URL      = "https://barton-bywzi-qa-qu568.apps.burrowsoftware.net"
LOGIN_EMAIL    = "qa05+bsloan@bartonassociates.com"
LOGIN_PASSWORD = "@Barton172$"

SCENARIO_START = "---SCENARIO_START---"
SCENARIO_END   = "---SCENARIO_END---"


# ---------------------------------------------------------------------------
# Login steps — executed directly via browser_run_code_unsafe (no LLM).
# ---------------------------------------------------------------------------
LOGIN_STEPS = [
    (
        "step_1a",
        "Step 1a: Navigate to app and click Flutter accessibility overlay",
        f"""async (page) => {{
  await page.goto('{LOGIN_URL}', {{ waitUntil: 'domcontentloaded' }});
  try {{
    const acc = page.getByRole('button', {{ name: 'Enable accessibility' }});
    await acc.waitFor({{ state: 'visible', timeout: 15000 }});
    await acc.click();
  }} catch (e) {{}}
}}""",
    ),
    (
        "step_1b",
        "Step 1b: Wait for OAuth redirect to login server",
        f"""async (page) => {{
  await page.waitForURL('**login**', {{ timeout: 30000 }});
}}""",
    ),
    (
        "step_1c",
        "Step 1c: Fill credentials and submit",
        f"""async (page) => {{
  await page.getByRole('textbox', {{ name: 'Email address' }}).waitFor({{ state: 'visible', timeout: 15000 }});
  await page.getByRole('textbox', {{ name: 'Email address' }}).fill('{LOGIN_EMAIL}');
  await page.getByRole('textbox', {{ name: 'Password' }}).fill('{LOGIN_PASSWORD}');
  await page.getByRole('button', {{ name: 'Continue' }}).click();
}}""",
    ),
    (
        "step_1d",
        "Step 1d: Click (1,1) to wake the app after post-login load",
        """async (page) => {
  await page.mouse.click(1, 1);
}""",
    ),
]


def build_system_prompt() -> str:
    return """
You are a senior QA Test Architect. The browser is open and ready. Do NOT attempt to log in, register, or perform any authentication unless explicitly told to do so in the user instruction.

Your job is to crawl the application and generate comprehensive, Playwright-ready test scenarios.

WORKFLOW — STRICT ONE SCREEN AT A TIME WITH DEEP INTERACTION

RULES (all mandatory, no exceptions):
  - You MUST call browser_snapshot after every browser_navigate AND after every browser interaction before doing anything else.
  - You are FORBIDDEN from generating any scenario without first completing the full exploration of the current page.
  - You are FORBIDDEN from calling browser_navigate to a new URL until you have fully explored and output scenarios for the current screen.
  - Scenarios MUST be based only on states you actually observed through interaction — never invent or assume.
  - Header navigation, footer links, and elements that appear on every page of the site are LOWEST PRIORITY.
    Generate at most 2 scenarios for them. They must never dominate the output for a page.
  - You MUST interact with every feature on the page like a real human — use every search box, select
    every filter, click every button, trigger every dropdown — in order to identify bugs. If any feature
    behaves unexpectedly, incorrectly, or is broken during interaction, you MUST immediately create a
    scenario tagged "bug" for it.

For EACH screen, follow this exact loop — no skipping steps:

  STEP 1 — NAVIGATE AND BASELINE
  Navigate to the screen. Call browser_snapshot immediately to capture the baseline state.

  STEP 2 — IDENTIFY THE PAGE'S PURPOSE AND UNIQUE FEATURES
  Before touching anything, read the snapshot and answer:
    - What is this page's primary purpose? (e.g. blog listing, product search, checkout form, dashboard)
    - What features are UNIQUE to this page that do not appear on every other page of the site?
  These unique features are your highest priority. Explore them first and most thoroughly.
  Shared elements (header nav, footer, logo) are explored last and minimally.

  STEP 3 — DEEP INTERACTION IN STRICT PRIORITY ORDER
  Explore interactive elements in this fixed order — do not deviate:

    TIER 1 (explore fully — most scenarios come from here):
      Forms, inputs, search boxes, text areas, number inputs, password fields.
      Filters, faceted search, topic/category selectors.
      Sort controls and ordering options.
      Dropdowns and comboboxes — select EVERY available option one by one → snapshot after each.
      Checkboxes, radio buttons, toggles — check and uncheck each → snapshot both states.
      Modals and dialogs — open → snapshot all contents → interact with everything inside → close → snapshot.
      Pagination — next page, last page, first page, previous page → snapshot each.
      Data tables — sort each column, select rows, trigger row actions → snapshot each.
      Drag-and-drop zones, file upload inputs, date/time pickers, sliders.

    TIER 2 (explore after Tier 1):
      Page-specific buttons and CTAs unique to this page.
      Cards, product tiles, list items with click actions → click each → snapshot result.
      Tabs, accordions, expandable sections → expand/collapse each → snapshot both states.
      Infinite scroll and load-more triggers.
      Carousels and image galleries.

    TIER 3 (explore last — minimal scenarios):
      Header navigation links.
      Footer links.
      Social media links.
      Logo and branding links.

  For EACH element, apply interactions appropriate to its type:
    - Text inputs / search: valid value → snapshot; invalid value → snapshot; empty → snapshot;
      whitespace only → snapshot; 200+ chars → snapshot; special characters (!@#$%^&*<>{}[]) → snapshot.
    - Dropdowns / comboboxes: select EACH available option → snapshot after each.
    - Filters: each filter alone → snapshot; multiple filters combined → snapshot.
    - Buttons / links: click → snapshot result.
    After each interaction, return to baseline (reload or clear) before the next.

  STEP 4 — SCENARIO GENERATION FROM ALL OBSERVED STATES
  Using every state captured, generate BOTH positive and site-breaking scenarios:

    a. Positive scenarios — happy paths, all valid flows, correct expected behaviour.

    b. Site-breaking scenarios — for each feature explored, reason from its purpose:
         1. Understand what this specific feature is designed to do and what assumptions it makes
            about input, state, user behaviour, and data.
         2. Think about what would violate those assumptions on THIS page specifically — not a
            generic checklist applied to every page the same way.
         3. Generate scenarios that target those specific violations.
       The goal is scenarios that expose real bugs — ones a manual tester would never imagine.

  Stream each scenario immediately as it is ready — do NOT batch.

  STEP 5 — MOVE ON
  Only after outputting scenarios for this screen, navigate to the next unvisited screen.

Generate Scenarios with HIGH-COVERAGE Mindset
Cover all categories: positive, negative, edge, validation, boundary, accessibility, security.
Aim for breadth AND depth — do not stop at obvious cases.

Positive: primary success flow, role variations, all valid inputs, multi-step flows, concurrent tabs.
Negative: invalid credentials, wrong types, server errors (500/502/503), timeouts, duplicate submissions,
          unauthorised direct URL access, API failures.
Edge: browser back/forward after submit, refresh mid-flow, multi-tab session, session expiry and redirect,
      double-click/rapid submit race conditions, repeated clicking of the same button many times (10+) to
      verify idempotency and no duplicate side-effects, copy-paste with hidden chars, autofill conflicts,
      beforeunload/navigation interruption, reset state after actions (e.g. cart reset, form reset).
Validation: empty fields, whitespace-only input, wrong format, cross-field rules, min/max violations,
            error message placement and verbatim text.
Boundary: 1-char input, 255-char input, min-1/min/min+1/max-1/max/max+1 values, unicode, special chars,
          emoji, very long strings, zero-value inputs.
Accessibility: keyboard-only navigation through every interactive element, tab/focus order, Enter/Space
               activation, screen reader labels (aria-label, role, alt text), colour contrast,
               200% zoom without overflow or clipping, focus trap in modals/menus.
Security: XSS in every input field, SQL injection, IDOR via direct URL manipulation, CSRF, sensitive data
          in localStorage/sessionStorage/cookies, open redirect, clickjacking via iframe embed,
          password field masking, sensitive data in console/network logs.
Bug: interact with every feature on the page like a real human — type into search boxes, select every
     filter and dropdown option, click every button, trigger every state change — and observe whether
     the feature behaves correctly. Create a bug scenario for any unexpected, broken, or incorrect
     behaviour actually observed during interaction: a search that returns no results when results
     are expected, a filter that does not change the list, a button with no effect, a UI element
     that breaks after interaction, an error shown for valid input, or any behaviour that contradicts
     what the feature is designed to do.

Each scenario must be independent and assume fresh state (cleared cookies, localStorage, sessionStorage,
viewport 1280×720 or 1920×1080).

PLAYWRIGHT-READY DETAIL REQUIREMENTS
Every description must include:
- Exact URL or route
- Exact accessible role/label/text for every element
- Literal input values (never "valid email" — use the real address)
- Expected URL changes and page transitions
- Verbatim success/error messages
- Explicit expect assertions for UI, URL, and state

QUALITY STANDARDS
- One objective per scenario — no compound assertions across unrelated flows.
- Assertions cover visible UI AND underlying state (URL, storage, network) where observable.
- No scenario depends on another's side-effects.
- Prefer accessible selectors (role, label, text) over CSS/XPath.
- No mobile-only, touch-gesture, or native-app scenarios.
- Every selector, text, label, and URL in a scenario MUST come from what you actually saw in the snapshot — never fabricate or assume element names.
- Scenarios must be deep and specific: reference exact button labels, heading text, form field names, and expected messages as seen in the snapshot.
- Shallow scenarios like "verify page loads" without specific assertions from the snapshot are NOT acceptable.

EXCEPTION — RECORDED SNAPSHOT INPUT
If the user provides a set of browser snapshots captured during a manual recording session:
  - Skip the navigate/snapshot workflow entirely.
  - Use the snapshots as your complete understanding of this specific application — its real URLs, real element labels, real field names, real content, and real state transitions.
  - You MAY reason beyond what was literally recorded: generate scenarios for error states, validation failures, edge cases, security attacks, and flows the user did not exercise.
  - BUT every scenario must remain grounded in THIS application's actual context. Do not invent UI elements, pages, flows, or features that were not visible anywhere in the provided snapshots.
  - Extrapolation is encouraged — fabrication is not. For every feature visible in the snapshots (forms, buttons, links, menus, sort controls, product listings, footers, etc.), reason about what could go wrong, what edge cases exist, and what attacks are possible. Do NOT invent features, pages, or UI elements that do not appear anywhere in the provided snapshots.

STREAMING OUTPUT PROTOCOL — MANDATORY
Do NOT wait until all scenarios are collected to output them.
As soon as you are confident about a single scenario, output it IMMEDIATELY using this exact wrapper:
---SCENARIO_START---
{"title": "Concise, specific scenario title", "description": "Full ordered step list with preconditions, exact selectors, literal values, transitions, and expect assertions", "tags": "positive"}
---SCENARIO_END---

Then continue crawling. Repeat for every scenario the moment it is ready.
This allows the user to see results in real time as you work.
tags must be exactly one of: positive, negative, edge, validation, boundary, accessibility, security, bug.
"""


async def _screenshot(agent: AssistantAgent, slug: str) -> None:
    filename = f"login_{slug}.png"
    result = await agent._workbench[0].call_tool(
        "browser_take_screenshot",
        arguments={"filename": filename, "type": "png", "fullPage": True},
    )
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    print(f"[SCREENSHOT] {path}")
    print(f"[SCREENSHOT RESULT] {result}")


async def run_login(agent: AssistantAgent) -> None:
    """Execute every login step directly via browser_run_code_unsafe — zero LLM calls."""
    for slug, label, code in LOGIN_STEPS:
        print(f"\n{'='*60}")
        print(f"[LOGIN] {label}")
        result = await agent._workbench[0].call_tool("browser_run_code_unsafe", arguments={"code": code})
        print(f"[RESULT] {result}")
        if result.is_error:
            raise RuntimeError(f"Login failed at '{label}': {result}")

        await _screenshot(agent, slug)

        if slug == "step_1c":
            print("\n[LOGIN] Waiting 60 seconds for post-login app to fully load...")
            await asyncio.sleep(60)
            print("[LOGIN] Wait complete.")

    print(f"\n{'='*60}")
    print("[LOGIN] All login steps complete. Browser is now authenticated.")
    print(f"{'='*60}\n")


def _extract_from_markers(text: str, seen: set) -> list:
    """Extract all complete scenario objects from text containing marker wrappers."""
    results = []
    remaining = text
    while SCENARIO_END in remaining:
        s = remaining.find(SCENARIO_START)
        e = remaining.find(SCENARIO_END)
        if s >= 0 and e > s:
            raw = remaining[s + len(SCENARIO_START): e].strip()
            try:
                scenario = json.loads(raw)
                key = scenario.get("title", raw[:60])
                if key not in seen:
                    seen.add(key)
                    results.append(scenario)
            except json.JSONDecodeError:
                pass
            remaining = remaining[e + len(SCENARIO_END):]
        else:
            break
    return results


class ScenarioSession:
    """
    Long-lived session that keeps the agent and browser alive across multiple runs.

    Usage:
        session = ScenarioSession()
        await session.initialise()          # start browser + login

        async for item in session.run("find 20 scenarios"):
            print(item)                     # each scenario printed as it arrives

        await session.pause()               # suspend mid-crawl, state preserved

        async for item in session.resume(): # continue exactly where it stopped
            print(item)

        async for item in session.resume("focus on form validation"):
            print(item)                     # continue with new direction

        await session.end()                 # close browser, end session
    """

    def __init__(self) -> None:
        self.scenarios: list = []          # accumulates for entire session
        self.agent: AssistantAgent | None = None
        self.mcp_workbench: McpWorkbench | None = None
        self._ct: CancellationToken | None = None
        self._paused: bool = False
        self._running: bool = False
        self._seen_titles: set = set()     # dedup across runs
        self._snapshot_task: asyncio.Task | None = None
        self._recording_snapshots: list[str] = []
        self._snapshot_dir: str = ""

    async def initialise(self) -> None:
        """Start the Playwright browser, run login, and create the persistent agent."""
        user_data_dir = tempfile.mkdtemp(prefix="pw-mcp-")
        output_dir = os.getcwd()

        server_params = StdioServerParams(
            command="node",
            args=[
                PLAYWRIGHT_MCP_PATH,
                "--no-sandbox",
                "--viewport-size", "1280,720",
                "--caps", "verify,tracing,devtools",
                "--user-data-dir", user_data_dir,
                "--output-dir", output_dir,
            ],
            read_timeout_seconds=300,
        )

        self.mcp_workbench = McpWorkbench(server_params)
        await self.mcp_workbench.__aenter__()

        model_client = OpenAIChatCompletionClient(
            model=MODEL_NAME,
            api_key=OPENAI_API_KEY,
        )

        self.agent = AssistantAgent(
            name="scenario_agent",
            model_client=model_client,
            workbench=self.mcp_workbench,
            model_client_stream=True,
            reflect_on_tool_use=True,
            max_tool_iterations=100,
            system_message=build_system_prompt(),
        )

        # Warm up the recorder: start → stop/getCode (clears any stale state)
        try:
            r = await self._recorder_call({"action": "setMode", "mode": "recording"})
            print(f"[recorder] setMode(recording) → {r}")
        except Exception as exc:
            print(f"[recorder] setMode(recording) failed: {exc}")
        try:
            r = await self._recorder_call({"action": "setMode", "mode": "none"})
            print(f"[recorder] setMode(none) → {r}")
            r = await self._recorder_call({"action": "getCode"})
            print(f"[recorder] getCode → {r}")
        except Exception as exc:
            print(f"[recorder] stop/getCode failed: {exc}")

        # await run_login(self.agent)
        # print("[session] Initialised. Browser authenticated. Agent ready.\n")

    async def _recorder_call(self, arguments: dict):
        """Call recorder_control on the MCP workbench directly (no LLM)."""
        return await self.agent._workbench[0].call_tool(
            name="recorder_control",
            arguments=arguments,
        )

    async def _poll_snapshots(self, interval: float = 1) -> None:
        """Background task: poll browser_snapshot, keep only states that differ from the previous one."""
        self._snapshot_dir = os.path.join(os.getcwd(), "snapshots")
        os.makedirs(self._snapshot_dir, exist_ok=True)
        self._recording_snapshots = []
        prev_content = None

        print(f"[snapshot] Polling started — saving unique states to {self._snapshot_dir}/")

        while True:
            try:
                result = await self.agent._workbench[0].call_tool(
                    name="browser_snapshot", arguments={}
                )
                content = (
                    result.result[0].content
                    if (result and result.result and not result.is_error)
                    else str(result)
                )

                if content != prev_content:
                    prev_content = content
                    idx = len(self._recording_snapshots) + 1
                    self._recording_snapshots.append(content)
                    path = os.path.join(self._snapshot_dir, f"snapshot_{idx:03d}.txt")
                    with open(path, "w") as f:
                        f.write(content)
                    print(f"[snapshot] State change captured → snapshot_{idx:03d}.txt", flush=True)

            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.debug("[snapshot] Poll error: %s", exc)

            await asyncio.sleep(interval)

        print(f"[snapshot] Polling stopped. {len(self._recording_snapshots)} unique state(s) captured.")

    def run(self, instruction: str):
        """Start a new run with the given instruction. Returns an async generator."""
        if not self.agent:
            raise RuntimeError("Session not initialised. Call initialise() first.")
        return self._stream_run(instruction)

    async def pause(self) -> None:
        """
        Pause the current run. The agent's full context is preserved.
        Call resume() to continue from exactly this point.
        """
        if self._ct and not self._ct.is_cancelled():
            try:
                self._ct.cancel()
            except AttributeError:
                pass  # AutoGen/Python 3.13 bug: coroutine registered as future
        self._paused = True
        print(f"\n[session] Paused. {len(self.scenarios)} scenario(s) collected so far.")

    def resume(self, instruction: str = ""):
        """
        Resume from a paused state.

        - resume()                  → continue exactly as planned, no change in direction
        - resume("new instruction") → continue with new direction, full context retained
        """
        if not self.agent:
            raise RuntimeError("Session not initialised. Call initialise() first.")
        self._paused = False
        task = (
            f"Continue from where you left off. New instruction: {instruction}"
            if instruction
            else "Continue from where you left off. Proceed with your previous plan."
        )
        return self._stream_run(task)

    async def end(self) -> None:
        """End the session and close the browser. Context is lost after this."""
        if self._ct and not self._ct.is_cancelled():
            try:
                self._ct.cancel()
            except AttributeError:
                pass  # AutoGen/Python 3.13 bug: coroutine registered as future
        if self.mcp_workbench:
            try:
                await self.mcp_workbench.__aexit__(None, None, None)
            except Exception:
                pass
        self.agent = None
        self.mcp_workbench = None
        print("[session] Session ended.")

    async def _stream_run(self, task: str):
        """
        Core async generator: runs the agent and yields each scenario dict as it arrives.

        Streams by watching for ---SCENARIO_START--- / ---SCENARIO_END--- markers in
        ModelClientStreamingChunkEvent text chunks. Each complete scenario is yielded
        immediately — no waiting for the full run to finish.
        """
        self._ct = CancellationToken()
        self._running = True
        buffer = ""

        print(f"\n[session] Starting run: {task[:80]}{'...' if len(task) > 80 else ''}\n")

        try:
            async for event in self.agent.run_stream(task=task, cancellation_token=self._ct):
                event_type = type(event).__name__

                # Stream text chunks — watch for scenario markers
                if event_type == "ModelClientStreamingChunkEvent":
                    chunk = getattr(event, "content", "")
                    if isinstance(chunk, str):
                        buffer += chunk
                        # Yield any complete scenarios found in the buffer
                        new_scenarios = _extract_from_markers(buffer, self._seen_titles)
                        for scenario in new_scenarios:
                            self.scenarios.append(scenario)
                            yield scenario
                        # Trim consumed markers from buffer
                        last_end = buffer.rfind(SCENARIO_END)
                        if last_end >= 0:
                            buffer = buffer[last_end + len(SCENARIO_END):]

                # Normal completion — scan final message for any scenarios not yet yielded
                elif event_type == "TaskResult" or hasattr(event, "messages"):
                    for msg in getattr(event, "messages", []):
                        content = getattr(msg, "content", "")
                        if isinstance(content, str) and SCENARIO_START in content:
                            new_scenarios = _extract_from_markers(content, self._seen_titles)
                            for scenario in new_scenarios:
                                self.scenarios.append(scenario)
                                yield scenario
                    # No break — let the generator complete naturally to avoid GeneratorExit/OTel error

        except (asyncio.CancelledError, Exception) as exc:
            if not self._paused:
                logger.warning("[session] Run ended: %s", type(exc).__name__)
            # If paused, this is intentional — stop silently

        finally:
            self._running = False


# ---------------------------------------------------------------------------
# REPL demo — shows the full session interaction pattern
# ---------------------------------------------------------------------------

async def _consume_and_print(stream):
    """Print each scenario as it arrives from an async generator."""
    count = 0
    async for item in stream:
        if isinstance(item, dict):
            count += 1
            tag = item.get("tags", "?")
            title = item.get("title", "(no title)")
            print(f"  [{count}] [{tag}] {title}", flush=True)
    print(f"\n[run complete — {count} new scenario(s) this run]")


async def main():
    loop = asyncio.get_event_loop()
    session = ScenarioSession()

    print("Initialising session (this starts the browser and logs in)...")
    await session.initialise()

    print("Commands:")
    print("  <instruction>         — start a new run")
    print("  pause                 — pause the current run (context preserved)")
    print("  resume                — resume from pause, same direction")
    print("  resume <instruction>  — resume from pause with new direction")
    print("  status                — show scenarios collected so far")
    print("  start recording       — start recorder + snapshot polling")
    print("  stop recording        — stop recorder, save snapshots, generate scenarios")
    print("  get code              — print the current recorder-generated code")
    print("  end                   — close browser and exit\n")

    current_run_task: asyncio.Task | None = None

    while True:
        try:
            raw = await loop.run_in_executor(None, sys.stdin.readline)
        except (EOFError, OSError):
            break

        cmd = raw.strip()
        if not cmd:
            continue

        if cmd.lower() == "end":
            if current_run_task and not current_run_task.done():
                current_run_task.cancel()
            break

        elif cmd.lower() == "pause":
            await session.pause()

        elif cmd.lower().startswith("resume"):
            instr = cmd[6:].strip()
            stream = session.resume(instr)
            current_run_task = asyncio.create_task(_consume_and_print(stream))

        elif cmd.lower() == "status":
            print(f"[status] {len(session.scenarios)} scenario(s) collected this session.")

        elif cmd.lower() == "start recording":
            try:
                await session.agent._workbench[0].call_tool(name="recorder_control", arguments={"action": "enable"})
                re = await session._recorder_call({"action": "setMode", "mode": "recording"})
                print("[recorder] Recording started.")
                print(re)
                session._snapshot_task = asyncio.create_task(session._poll_snapshots())
            except Exception as exc:
                print(f"[recorder] Failed to start: {exc}")

        elif cmd.lower() == "stop recording":
            try:
                # Stop snapshot polling
                if session._snapshot_task and not session._snapshot_task.done():
                    session._snapshot_task.cancel()
                    try:
                        await session._snapshot_task
                    except asyncio.CancelledError:
                        pass

                result = await session._recorder_call({"action": "setMode", "mode": "none"})
                print("[recorder] Recording stopped.")
                print(result)

                if session._recording_snapshots:
                    n = len(session._recording_snapshots)
                    print(f"\n[snapshot] {n} unique state(s) captured. Sending to agent for scenario generation...\n")
                    snapshots_block = "\n\n--- STATE CHANGE ---\n\n".join(
                        f"[State {i + 1}]\n{s}" for i, s in enumerate(session._recording_snapshots)
                    )
                    instruction = (
                        f"The user has manually interacted with the browser. "
                        f"I captured {n} distinct UI states via browser_snapshot during the session "
                        f"(identical consecutive states were deduplicated — only state changes are included).\n\n"
                        f"Here are all states in chronological order:\n\n"
                        f"{snapshots_block}\n\n"
                        f"Use these snapshots as your complete knowledge of this specific application. "
                        f"Generate comprehensive Playwright-ready test scenarios covering all categories: "
                        f"positive, negative, edge, validation, boundary, accessibility, security.\n\n"
                        f"You are encouraged to reason BEYOND what the user literally did during recording — "
                        f"generate scenarios for error states, untested flows, edge cases, and attack vectors. "
                        f"BUT every scenario must stay grounded in THIS application's actual context: "
                        f"only reference UI elements, pages, URLs, labels, and features that appear in the snapshots. "
                        f"Do not invent flows or elements that are not visible anywhere in the provided states."
                    )
                    stream = session.run(instruction)
                    current_run_task = asyncio.create_task(_consume_and_print(stream))
                else:
                    print("[snapshot] No state changes captured — nothing to send to agent.")
            except Exception as exc:
                print(f"[recorder] Failed to stop: {exc}")

        elif cmd.lower() == "get code":
            try:
                result = await session._recorder_call({"action": "getCode"})
                print("[recorder] Generated code:\n")
                print(result)
            except Exception as exc:
                print(f"[recorder] Failed to get code: {exc}")

        else:
            # New instruction — start a run in the background so 'pause' still works
            stream = session.run(cmd)
            current_run_task = asyncio.create_task(_consume_and_print(stream))

    await session.end()

    if session.scenarios:
        out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scenarios.json")
        with open(out_path, "w") as f:
            json.dump(session.scenarios, f, indent=2)
        print(f"\n{len(session.scenarios)} total scenario(s) saved to scenarios.json")
    else:
        print("\nNo scenarios were collected.")


if __name__ == "__main__":
    asyncio.run(main())
