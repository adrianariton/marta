from core.attacks.agents import StatelessSelfHostedVLLMAgent
from core.attacks.evaluators import AgentPEC7Evaluator, PEC7, DummyPEC7Evaluator
from core.attacks.interfaces import LinearHistory, OneShotConversation, Conversation
from core.attacks.utils import get_faulty_json_data
EVALUATOR_DEFS = {
    "phi-4-AWQ":                       (8001, "/models/phi-4-AWQ",                       ["<|end|>", "<|endoftext|>"]),
    "Mixtral-8x22B-Instruct-v0.1-AWQ":(8002, "/models/Mixtral-8x22B-Instruct-v0.1-AWQ",["</s>"]),
    "Qwen2.5-32B-Instruct-AWQ":       (8003, "/models/Qwen2.5-32B-Instruct-AWQ",        ["<|im_end|>", "<|endoftext|>"]),
    "Qwen2.5-72B-Instruct-AWQ":       (8014, "/models/Qwen2.5-72B-Instruct-AWQ",        ["<|im_end|>", "<|endoftext|>"]),
}
from core.attacks.agents.batch import Batch
batches = [
    Batch(max_wait_seconds=3, max_batch_size=90, name=f"VLLM_{8001 + i}")
    for i in range(len(EVALUATOR_DEFS))
]

agents = {}
for i, (name, (port, model_path, stop_tokens)) in enumerate(EVALUATOR_DEFS.items()):
    if "mixtral" in name.lower():
        structured_outputs = None
    else:
        structured_outputs = {
            "json": {
                "type": "object",
                "properties": {
                    "level": {"type": "number"},
                },
                "required": ["level"],
                "additionalProperties": False,
            }
        }
    agents[name] = StatelessSelfHostedVLLMAgent(
        host="localhost",
        port=port,
        model=model_path,
        temperature=0.7,
        max_tokens=1024,
        stop=stop_tokens + ["User:", "Assistant:"],
        presence_penalty=0.0,
        repetition_penalty=1.1,
        frequency_penalty=0.0,
        top_p=0.9,
        structured_outputs=structured_outputs,
        default_system_message=not ("mixtral" in name.lower())
    )
    


class Mapper:
    def __init__(self, agent: StatelessSelfHostedVLLMAgent, map_size: tuple[int, int]=None):
        assert isinstance(agent, StatelessSelfHostedVLLMAgent)
        self.agent = agent
        self.map_size = map_size or (100,100)
    
    async def get_pixel(self, xs: list[int], ys: list[int]):
        convos = []
        for x, y in zip(xs, ys):
            prompt = f"You are in a 100x100 world ar position ({x}, {y}). What is the height level w.r.t the water level? Please provide json that contains a single parameter: level, which is a number between 0 and 100 signifying the height level w.r.t the water level."
            convo = Conversation()
            # convo.set_system_message("You are a helpful assistant that only responds with jsons of this format  \{'level': <0-100>\}")
            convo.add_text_message(prompt, 'user')
            convos.append(convo.latest_oneshot())
        try:
            pixel_results = await self.agent.abatch_generate(convos=convos)
            pixel_levels = [get_faulty_json_data(p, 'level') or -1.0 for p in pixel_results]
            # print(pixel_levels)
            pixel_levels = [x if isinstance(x, int) or isinstance(x, float) else 0.0 for x in pixel_levels]
            return pixel_levels
        except Exception as e:
            print(f'{e}')
            return [-1] * len(xs)
        
    def __repr__(self):
        return str((self.agent)) + "::mapper"
    def __str__(self):
        return str((self.agent)) + "::mapper"
        

MAP_SIZE = 50
CHUNK_SIZE = 32
print(list(enumerate(agents)))
mappers = {x: Mapper(y, map_size=(MAP_SIZE, MAP_SIZE)) for x, y in agents.items()}
print(mappers)
OUTPUT_PATH = "map.png"
CONCURRENCY = 10
# Color palettes per mapper: (low_color_rgb, high_color_rgb)
MAPPER_PALETTES = [
    ((0, 0, 205),   (0, 139, 0)),    # 0: blue → green (ocean/land)
    # ((139, 0, 0),   (255, 165, 0)),  # 1: dark red → orange (volcanic)
    # ((20, 20, 40),  (220, 220, 255)),# 2: near-black → pale lavender (lunar)
    # ((0, 60, 60),   (180, 255, 180)),# 3: deep teal → mint (alien)
    # ((101, 67, 33), (255, 250, 205)),# 4: brown → light yellow (desert)
]

def level_to_rgb(level: float, palette_idx: int) -> tuple[int, int, int]:
    """Interpolate between low and high color for a given palette."""
    low, high = MAPPER_PALETTES[palette_idx % len(MAPPER_PALETTES)]
    t = max(0.0, min(1.0, level / 100.0))
    r = int(low[0] + (high[0] - low[0]) * t)
    g = int(low[1] + (high[1] - low[1]) * t)
    b = int(low[2] + (high[2] - low[2]) * t)
    return r, g, b
import asyncio
import numpy as np
from PIL import Image

from tqdm.asyncio import tqdm

async def main():
    xy_pairs = [(x, y) for y in range(MAP_SIZE) for x in range(MAP_SIZE)]
    chunks = [xy_pairs[i:i+CHUNK_SIZE] for i in range(0, len(xy_pairs), CHUNK_SIZE)]

    mapper_list = list(mappers.values())
    n_mappers = len(mapper_list)
    sem = asyncio.Semaphore(CONCURRENCY)

    async def throttled_get_pixel(mapper, xs, ys):
        async with sem:
            return await mapper.get_pixel(xs, ys)

    all_tasks = []
    all_meta = []
    for mapper_idx, mapper in enumerate(mapper_list):
        for chunk in chunks:
            xs = [p[0] for p in chunk]
            ys = [p[1] for p in chunk]
            all_tasks.append(throttled_get_pixel(mapper, xs, ys))
            all_meta.append((list(zip(xs, ys)), mapper_idx))

    results = await tqdm.gather(*all_tasks, desc="Fetching chunks", unit="chunk")

    rgb_maps = [np.zeros((MAP_SIZE, MAP_SIZE, 3), dtype=np.uint8) for _ in range(n_mappers)]

    for (coords, mapper_idx), levels in tqdm(
        zip(all_meta, results), total=len(all_meta), desc="Painting pixels", unit="chunk"
    ):
        for (x, y), level in zip(coords, levels):
            effective_level = level if level >= 0 else 0
            rgb_maps[mapper_idx][y][x] = level_to_rgb(effective_level, mapper_idx)

    for mapper_idx, rgb in enumerate(rgb_maps):
        path = f"map_mapper_{mapper_idx}.png"
        img = Image.fromarray(rgb, mode='RGB')
        img = img.resize((500, 500), Image.Resampling.NEAREST)
        img.save(path)
        print(f"Saved {path}")

    print("\nDone!")
    

if __name__ == "__main__":
    asyncio.run(main())