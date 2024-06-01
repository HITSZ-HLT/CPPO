
### Training Process


1. Train SFT:
    ```bash
    cd sft/ && deepspeed train_sft1_xl.py
    ```

2. Train Reward Model:
    ```bash
    cd reward_model/ && deepspeed train_rm_Data1.py
    ```

3. CPPO-Heuristic training:
    ```bash
    accelerate launch --config_file configs/default_accelerate_config.yaml cppoH_gpt2s_task-1.py
    ```
 
4. CPPO-Learn training:
    ```bash
    accelerate launch --config_file configs/default_accelerate_config.yaml cppoL_gpt2s_task-1.py
    ```

5. Evaluate on reference PM
    ```bash
    python trlx_inference_part.py --datapart Data1
    ```

## References

1. Nisan Stiennon, Long Ouyang, Jeff Wu, Daniel M. Ziegler, Ryan Lowe, Chelsea Voss, Alec Radford, Dario Amodei, Paul Christiano, "[Learning to Summarize from human feedback](https://arxiv.org/abs/2009.01325)", Neural Information Processing Systems, 2020.
