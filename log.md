# 1. We need NCCL to know that our algorithm exists

## `src/include/nccl_common.h`

a. increment `NCCL_NUM_ALGORITHMS`

b. add a macro for your algorithm

## `src/init.cc`

c. add an entry in `ncclAlgoStr`

d. update `graphs` in `initTransportsRank` to have a nonnull entry at your algorithm's index. Note: I currently don't know what you will have to do if your algorithm is sufficiently different from ring/tree.

## `src/enqueue.cc`

e. edit `updateCollCostTable` such that `table[a][p]` is set to not-(-1.0) so that NCCL doesn't think your algo is invalid (e.g. just set it to 0.0).

## `src/device/generate.py`

f. add your algo name to `all_algos` and in `algos_of_coll` put your algorithm under its corresponding collectives.

## `src/include/device.h` 

g. edit `ncclDevFuncId` by incrementing `nAlgos` for the collectives that correspond to your algorithm (should reflect changes in part 1f) and add a stage to the ternary expression chain to handle your algorithm.

## `src/device/<coll>.h`

. create template instances for `RunWorkColl`

# 2. NCCL needs to know how to connect GPUs to one another for your algorithm

## `src/include/proxy.h`

a. add a value in the `ncclPattern_t` enum

## `src/include/transport.h`

b. add a `ncclTransport<algo>Connect` definition

## `src/enqueue.cc`

c. in `calcCollChunking`, update the `switch (info->func)` statement to set the pattern to the value from 2a for the relevant collectives.

d. still in `calcCollChunking`, update `switch (pattern)` statement to prevent NCCL from throwing an exception.

## `src/proxy.cc`

e. update `NeedProxy` to handle your pattern

f. in `ncclProxySaveOp`, add a case to the `switch (op->pattern)`

## `src/transport/generic.cc`

g. write a definition for 2b. Note, when you call `ncclTransportP2pSetup`, the index into `comm->graphs` is based on how you set `graphs` in 1d
