from core.attacks.agents import MistralAgent, LlamaAgent
import pandas as pd
from core.attacks.classifiers.refusal import HeuristicRefusalClassifier
from core.attacks.interfaces import Batch, OneShotConversation
import asyncio

mistral_batch = Batch(max_batch_size=20, max_wait_seconds=5, check_frequency=1.0)
llama_batch = Batch(max_batch_size=20, max_wait_seconds=5, check_frequency=1.0)

ma = MistralAgent(max_new_tokens=1024).with_batch(mistral_batch)
la = LlamaAgent(max_new_tokens=1024).with_batch(llama_batch)


def get_target(ag):
    if ag == "GraySwanAI/Mistral-7B-Instruct-RR":
        return ma
    elif ag == "Youliang/llama3-8b-derta":
        return la


async def process_one_row(row):
    if row["harmful"] == False or row["is_final"] == False:
        return None
    rref = HeuristicRefusalClassifier()
    if row["conversation"] is None:
        return None
    else:
        convo = OneShotConversation.from_default(row["conversation"])
    agent_reply = row["response"]
    if row["response"] is None or row["goal"] is None:
        return None
    convo.add_text_message(agent_reply, "assistant")
    goal = row["goal"]
    convo.add_text_message(goal, "user")
    agent = get_target(row["target"])
    # print(f"[begin] {row['conversation_id']}::{row['dataset_row_id']}::{row['goal']}")
    result = (await agent.agenerate(convo))[0]

    is_Ref = await rref.ais_refused(result)

    print(
        f"{row['conversation_id']}::{row['dataset_row_id']}::{row['goal']} -> {is_Ref=}", flush=True
    )

    if is_Ref:
        return {"response_to_goal": result, "ref": is_Ref}

    return {"response_to_goal": result, "ref": is_Ref}


PARALLEL = 20


async def loop_over(df):
    sem = asyncio.Semaphore(PARALLEL)  # Now we limit the tasks

    async def sem_task(row):
        async with sem:
            return await process_one_row(row)

    # Create the list of tasks using the semaphore wrapper
    tasks = [sem_task(row) for _, row in df.iterrows()]
    return await asyncio.gather(*tasks)


df = pd.read_parquet("alternative_huge_data.parquet")
print(f"{len(df[(df['harmful'] == True) & (df['is_final'] == True)])} rows to go over.", flush=True)
df["goal_response_after_convo"] = asyncio.run(loop_over(df))
df.to_parquet("alternative_huge_data_goal.parquet")
print("done!!!!!!", flush=True)
