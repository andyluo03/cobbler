"""Entry-point for megakernel implementation. Currently just scaffolding"""

class Instruction:
    """
    These are the opcode instructions that each worker will be executing.
    Really, each instruction is backed by an entire kernel impl (e.g. Attn, FusedRMSNorm)
    """
    ...

class Schedule:
    """
    Controls how each instruction will be allocated per SM along a spatial and temporal axis.

    This could take in a hard-coded instruction stream and emit a set of "kernel calls".
    The MK itself should handle orchestration, but the instruction stream construction and
    partitioning across SMs can be handled here AOT.

    The main thing here that's interesting to me coul be investigating inbalanced compute,
    which really necessitates you simulating different block/wave sizes across a varying number of SMs.
    This is after the basic scheduling problems and impl.
    """
    ...


### Needs to be in CUDA or Triton or something
class CompLoop:
    """
    The way MK works is each specialized kernel has an implementation of a particular instruction (defaults to NOOP),
    and it simply executes that instruction. This is done persistently in a loop.

    This will be waiting on some SM flag or control flag signaling an input has loaded (most likely the former, async is good),
    and do some computation. Typically these are as fused as possible.
    """
    ...


class CommLoop:
    """
    These can be split into two classes: Producer and Consumer. Depending on the type of comm being done,
    there are some nuances on the implementation of the instructions (I believe), though there is some unified behavior as well.
    """
    ...

class OrchestratorLoop:
    """
    My understanding is that this kernel is here for explicit synchronization across blocks, as well as
    handling fine-grained setup/teardown of sync buffers, semaphores, etc.
    """
    ...

class MKEntry:
    """
    MK entry-point. Handle setting up of the paged SM, etc.
    """
    ...
