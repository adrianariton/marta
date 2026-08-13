import json
import uuid
import asyncio
import requests
import aiohttp
from typing import Any, Optional
from cker.hf import load_hf_model
from cker.program import Program, Constant, assign, new_var, If, Else, ElseIf, While
from cker.hf import load_hf_model
from cker.packages import load_torch
from cker.hooks import AssignHook, ScopeHook

SERVER_URL = "http://localhost:8888"


ah = AssignHook()
sh = ScopeHook()


def main():
    with Program(SERVER_URL) as program:
        ah.start()
        sh.start()
        torch = load_torch()
        gpu_rand_tensor = new_var("gpu_rand_tensor")
        result = new_var("result")
        nr_imp = new_var("nr_imp")
        c = new_var("c")
        gpu_rand_tensor = torch.rand(2, 3, device=torch.device("cuda"))
        random_num = torch.rand(1).item()
        if True:
            result = 2 * gpu_rand_tensor[0:1, 0:1]
        result = 0
        c = 5
        nr_imp = []
        while c > 0:
            result += 1
            c -= 1
            if c % 2 == 1:
                nr_imp.append(c)
        # result = gpu_rand_tensor
        x = program.get(nr_imp)
        ah.end()
        sh.end()
        print("Result:", x)
        # hf_model = load_hf_model("gpt2")
        # output = hf_model.generate(
        #     prompt="How are you?",
        #     max_new_tokens=50,
        #     temperature=0.7,
        # )
        # text = output.text.strip().lower()
        # if text[0] == "i":
        #     text = "dsds"

        # print(program.debug())
        # sh.end()
        # result = program.get(text)
        # print("Result:", result)
        # ah.end()


if __name__ == "__main__":
    main()
