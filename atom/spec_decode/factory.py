from atom.config import Config
from atom.spec_decode.drafter import Drafter
from atom.spec_decode.dspark_proposer import DSparkProposer
from atom.spec_decode.eagle4_proposer import EAGLE4_ARCHITECTURES, Eagle4Proposer
from atom.spec_decode.eagle_proposer import EagleProposer


def build_drafter(config: Config, device, runner) -> Drafter:
    """Construct the speculative drafter for this config.

    Block-parallel DSpark vs serial EAGLE/MTP is the only flavor branch; every
    downstream call-site depends on the ``Drafter`` contract, not the concrete
    type.
    """
    if config.speculative_config.use_dspark():
        return DSparkProposer(config, device, runner)
    draft_arch = (
        config.speculative_config.draft_model_hf_config.architectures or [""]
    )[0]
    if draft_arch in EAGLE4_ARCHITECTURES:
        return Eagle4Proposer(config, device, runner)
    return EagleProposer(config, device, runner)
