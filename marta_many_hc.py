#!/usr/bin/env python3
import httpx
import asyncio

MODELS = {
    8001: "phi-4-AWQ",
    8002: "Mixtral-8x22B-Instruct-v0.1-AWQ",
    8003: "Qwen2.5-32B-Instruct-AWQ",
    8014: "Qwen2.5-72B-Instruct-AWQ",
}

async def check(client: httpx.AsyncClient, port: int, name: str):
    url = f"http://localhost:{port}"

    # /health
    try:
        r = await client.get(f"{url}/health", timeout=5)
        r.raise_for_status()
        health = "✅"
    except httpx.TimeoutException:
        print(f"  ⏳ TIMEOUT  :{port}  {name:<35}  /health no response within 5s")
        return
    except Exception as e:
        print(f"  ❌ DOWN     :{port}  {name:<35}  /health: {e}")
        return

    # /v1/models — retry a few times since it lags behind /health
    models_detail = "not ready yet"
    for attempt in range(5):
        try:
            m = await client.get(f"{url}/v1/models", timeout=5)
            m.raise_for_status()
            loaded = [x["id"] for x in m.json().get("data", [])]
            models_detail = f"serving: {', '.join(loaded)}"
            break
        except Exception:
            if attempt < 4:
                await asyncio.sleep(3)

    print(f"  {health} UP      :{port}  {name:<35}  {models_detail}")

async def main():
    print("\n── vLLM health check ──────────────────────────────────────────")
    async with httpx.AsyncClient() as client:
        await asyncio.gather(*[check(client, port, name) for port, name in MODELS.items()])
    print("───────────────────────────────────────────────────────────────\n")

if __name__ == "__main__":
    asyncio.run(main())