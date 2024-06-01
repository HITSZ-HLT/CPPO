import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def convert(pt_path,save_dir, name):
    model = AutoModelForCausalLM.from_pretrained(f"/userhome/Research_HUB/RLHF/trlx/examples/summarize_rlhf/download/{name}")
    chk = torch.load(pt_path)
    state_dict = model.state_dict()
    for n in state_dict:
        n2 = f"base_model.{n}"
        if n2 in chk['module']:
            state_dict[n] = chk['module'][n2]
        else:
            print(f"{n} is not loaded")

    s1 = model.transformer.wte.weight.data.std()
    model.load_state_dict(state_dict)
    s2 = model.transformer.wte.weight.data.std()

    print(f"\n Model load from {pt_path} \t"
          f"success = {(s1 != s2).item()}\n "
          f"global_steps={chk['global_steps']} \n")
    model.save_pretrained(save_dir)
    return model

pt_path = "mp_rank_00_model_states.pt"
save_dir = "./"
name="gpt2-s"

convert(pt_path,save_dir,name)