"""
example_memory_agent.py — How to use LongShortTermMemoryStorage with an ag2-beta Agent
========================================================================================

This file is a self-contained, runnable tutorial.  Set your API key and run:

    python example_memory_agent.py

What the example demonstrates
------------------------------
1. Creating LongShortTermMemoryStorage with just a model config — no custom
   summarizer or consolidator needed; built-in defaults handle both.
2. Wiring the storage into a MemoryStream.
3. Giving the agent a memory_lookup tool so it can drill into stored blocks.
4. Running a multi-turn conversation inside storage.session() so that
   L2 consolidation and JSON persistence happen automatically at session end.
5. Starting a second session that picks up the persisted L2 blocks and is
   forced to call memory_lookup to retrieve specific details.
"""

import asyncio
import os
from pathlib import Path

from autogen.beta.agent import Agent
from autogen.beta.config.openai.config import OpenAIConfig
from autogen.beta.history import LongShortTermMemoryStorage
from autogen.beta.stream import MemoryStream

# ---------------------------------------------------------------------------
# 0.  Configuration
# ---------------------------------------------------------------------------

# Paste your key here or export OPENAI_API_KEY before running.
API_KEY = os.environ.get("OPENAI_API_KEY", "None")

# Where to persist memory blocks between runs.
L2_STORE = Path("agent_memory.json")

# The model used for both the main agent and memory summarisation.
MODEL = "gpt-5-mini"

config = OpenAIConfig(model=MODEL, api_key=API_KEY)

# ---------------------------------------------------------------------------
# 1.  Create the storage
#     Passing only `config` is enough — LongShortTermMemoryStorage builds
#     a default summarizer and a smart LLM-driven consolidator automatically.
#     Override them with explicit `summarizer=` / `consolidator=` kwargs only
#     when you need custom behaviour.
# ---------------------------------------------------------------------------

storage = LongShortTermMemoryStorage(
    config=config,
    l2_store_path=L2_STORE,    # persists memory to disk between runs
    token_threshold=10_000,    # compress to L1 when raw context exceeds ~10k tokens
    max_l2_blocks=20,          # hard cap on long-term blocks
    recent_turns=5,            # keep last 5 turns as raw context
)


# ---------------------------------------------------------------------------
# 2.  Create the agent
#     We pass storage.memory_lookup_tool() so the agent can call
#     memory_lookup(block_id) to drill into the source text of any L1 block
#     or the sub-blocks of any L2 block.
# ---------------------------------------------------------------------------

agent = Agent(
    "assistant",
    prompt=(
        "You are a helpful assistant with long-term memory. "
        "Your memory context is pinned at the top of each message. "
        "Use memory_lookup(block_id) whenever you want to read the full "
        "detail behind a memory block."
    ),
    config=config,
    tools=[storage.memory_lookup_tool()],
)


# ---------------------------------------------------------------------------
# 3.  Session helper — wraps a conversation inside storage.session() so that
#     L2 consolidation runs automatically when the session ends.
# ---------------------------------------------------------------------------

async def run_diagnostic_session() -> None:
    """Send a single diagnostic prompt that asks the LLM to list every tool
    and memory block visible to it, then discard the conversation entirely.

    This session is intentionally NOT wrapped in ``storage.session()``, so:
      • No L1 compression is triggered (the diagnostic turn is short).
      • No L2 consolidation happens.
      • The raw events are dropped via ``storage.drop_history()`` at the end,
        leaving the persistent memory state completely unchanged.

    Use this between sessions to verify that the memory context looks correct.
    """
    stream = MemoryStream(storage=storage)

    print(f"\n{'='*60}")
    print("Diagnostic session  (not stored in long-term memory)")
    print('='*60)

    prompt = (
        "This is a diagnostic check. Please list:\n"
        "1. Every tool you currently have access to (name and one-line description).\n"
        "2. Every memory block visible in your context right now — "
        "   for each block state its ID, tier (L1 or L2), creation time, and abstract.\n"
        "Be exhaustive and format the output clearly."
    )

    print(f"\nUser: {prompt}")
    reply = await agent.ask(prompt, stream=stream)
    print(f"\nAgent: {await reply.content()}")

    # Discard the raw events for this stream so nothing leaks into L1/L2.
    await storage.drop_history(stream.id)
    print("\n[diagnostic] Raw events discarded — memory state unchanged.")


async def run_session(task: str, follow_ups: list[str] | None = None) -> None:
    """
    Run a user task (and optional follow-up tasks) in a single agent session.

    Each string is a complete, self-contained task description passed to
    agent.ask().  The agent may internally make multiple tool calls and
    LLM turns to complete it — all of that stays on the same stream.

    A new MemoryStream is created for each session so the raw event log
    starts fresh, but L1/L2 blocks from previous sessions are already
    loaded into `storage` (L2 from disk).
    """
    stream = MemoryStream(storage=storage)

    print(f"\n{'='*60}")
    print("Starting new session")
    if storage._l2_blocks:
        print(f"  Loaded {len(storage._l2_blocks)} L2 block(s) from {L2_STORE}")
    print('='*60)

    async with storage.session():           # <-- consolidate() called on exit
        # Primary task
        print(f"\nUser: {task}")
        reply = await agent.ask(task, stream=stream)
        print(f"Agent: {await reply.content()}")

        # Optional follow-up tasks — each continues on the same stream,
        # so the agent retains the full conversation context from above.
        for follow_up in (follow_ups or []):
            print(f"\nUser: {follow_up}")
            reply = await reply.ask(follow_up)   # continues the same context
            print(f"Agent: {await reply.content()}")

    # After the session: report what memory looks like now
    print(f"\n--- Memory state after session ---")
    for sid, blocks in storage._l1_blocks.items():
        print(f"  L1 blocks ({len(blocks)}): {[b.id for b in blocks]}")
    print(f"  L2 blocks ({len(storage._l2_blocks)}): {[b.id for b in storage._l2_blocks]}")
    if storage._l2_blocks:
        print(f"  Saved to: {L2_STORE}")


# ---------------------------------------------------------------------------
# 4.  Main — two sessions to show cross-session memory persistence
# ---------------------------------------------------------------------------

async def main() -> None:
    # ── Session 1 ────────────────────────────────────────────────────────
    # The user gives the agent a full task description in one shot.
    # If the conversation grows past token_threshold during the session,
    # older turns are compressed to L1 blocks automatically.
    # When the session ends, L1 blocks are consolidated into L2 and
    # written to agent_memory.json.
    await run_session(
        task=(
            "I'm Alex. I'm building a Python recommendation system that uses "
            "collaborative filtering with matrix factorisation. The main challenge "
            "is cold-start for new users; I'm exploring content-based fallback "
            "with item embeddings as a solution. "
            "Please give me a concise technical overview of this design and "
            "suggest one concrete improvement I could make."
        ),
        follow_ups=[
            # A genuine follow-up: a new, complete question that builds on the reply.
            "The cold-start suggestion sounds promising. "
            "Can you show me a minimal Python snippet implementing it?",
        ],
    )

    # ── Diagnostic ───────────────────────────────────────────────────────
    # Ask the agent to enumerate every tool and memory block it can see.
    # Nothing produced here is saved; the conversation is dropped on exit.
    #await run_diagnostic_session()

    # ── Session 2 ────────────────────────────────────────────────────────
    # A fresh stream, but storage already holds the L2 blocks persisted
    # from session 1.  The agent sees them in the pinned memory header.
    #
    # The task deliberately asks for the *exact* Python snippet that was
    # written in session 1.  That level of detail lives in an L1 source
    # block, NOT in the compressed L2 abstract, so the agent must call
    # memory_lookup(L2-id) to locate the relevant L1 IDs, then call
    # memory_lookup(L1-id) to retrieve the actual code.
    await run_session(
        task=(
            "Hi again. In our last conversation you showed me a Python snippet "
            "for the content-based cold-start fallback. "
            "Please look it up from your memory (use memory_lookup as needed for multiple times since the code should be a L1 block) "
            "and paste the exact code here."
        ),
        follow_ups=[
            # Builds on the retrieved code — forces the agent to reason about
            # the specific implementation it just looked up.
            "Good. Now extend that snippet to handle the edge case where "
            "no item embeddings are available at all (completely new item). "
            "Show only the changed/added lines."
        ],
    )


if __name__ == "__main__":
    asyncio.run(main())
