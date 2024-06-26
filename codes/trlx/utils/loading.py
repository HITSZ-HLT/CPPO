from typing import Callable

# Register load orchestrators via module import
from trlx.orchestrator import _ORCH
from trlx.orchestrator.offline_orchestrator import OfflineOrchestrator
from trlx.orchestrator.ppo_orchestrator import PPOOrchestrator
# from trlx.orchestrator.clppo_orchestrator import CLPPOOrchestrator
from trlx.orchestrator.dppo_orchestrator import DPPOOrchestrator
from trlx.orchestrator.hppo_orchestrator import HPPOOrchestrator
# from trlx.orchestrator.masppo_orchestrator import MASPPOOrchestrator
# from trlx.orchestrator.tfclppo_orchestrator import TFCLPPOOrchestrator
# from trlx.orchestrator.nlpo_orchestrator import NLPOOrchestrator

# Register load pipelines via module import
from trlx.pipeline import _DATAPIPELINE
from trlx.pipeline.offline_pipeline import PromptPipeline

# Register load trainers via module import
from trlx.trainer import _TRAINERS, register_trainer
from trlx.trainer.accelerate_ilql_trainer import AccelerateILQLTrainer
from trlx.trainer.accelerate_ppo_trainer import AcceleratePPOTrainer
from trlx.trainer.accelerate_sft_trainer import AccelerateSFTTrainer
# from trlx.trainer.accelerate_clppo_trainer import AccelerateCLPPOTrainer
from trlx.trainer.accelerate_dppo_trainer import AccelerateDPPOTrainer
from trlx.trainer.accelerate_hppo_trainer import AccelerateHPPOTrainer
# from trlx.trainer.accelerate_masppo_trainer import AccelerateMASPPOTrainer
# from trlx.trainer.accelerate_tfclppo_trainer import AccelerateTFCLPPOTrainer
# from trlx.trainer.accelerate_nlpo_trainer import AccelerateNLPOTrainer


try:
    from trlx.trainer.nemo_ilql_trainer import NeMoILQLTrainer
except ImportError:
    # NeMo is not installed
    def _trainer_unavailble(name):
        def log_error(*args, **kwargs):
            raise ImportError(f"Unable to import NeMo so {name} is unavailable")

        return register_trainer(name)(log_error)

    _trainer_unavailble("NeMoILQLTrainer")


def get_trainer(name: str) -> Callable:
    """
    Return constructor for specified RL model trainer
    """
    name = name.lower()
    if name in _TRAINERS:
        return _TRAINERS[name]
    else:
        raise Exception("Error: Trying to access a trainer that has not been registered")


def get_pipeline(name: str) -> Callable:
    """
    Return constructor for specified pipeline
    """
    name = name.lower()
    if name in _DATAPIPELINE:
        return _DATAPIPELINE[name]
    else:
        raise Exception("Error: Trying to access a pipeline that has not been registered")


def get_orchestrator(name: str) -> Callable:
    """
    Return constructor for specified orchestrator
    """
    name = name.lower()
    if name in _ORCH:
        return _ORCH[name]
    else:
        raise Exception("Error: Trying to access an orchestrator that has not been registered")
