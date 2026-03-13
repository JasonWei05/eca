# Ray Trainer Workflow

This document explains what happens when you run `dapo_eca/eca_test.sh` or another RL training script in this repo.

It is focused on the FSDP/FSDP2 + Ray path, because that is what `eca_test.sh` uses.

## Scope

This explains:

- how the shell script launches the job,
- how `TaskRunner` builds the trainer,
- how `RayPPOTrainer` or `RayDAPOTrainer` creates worker groups,
- how the FSDP worker contains the actor, rollout engine, and reference policy,
- how a single training step calls into rollout, reward, critic, ref, and actor update code.

This does not try to document every feature flag in the trainer.

## The Most Important Distinction

There are several things called "actor" in this stack. They are not the same.

- Ray worker: a Ray actor process scheduled by Ray.
- FSDP worker: `ActorRolloutRefWorker` or `AsyncActorRolloutRefWorker` inside that Ray process.
- PPO actor implementation: `DataParallelPPOActor`, a local Python object created inside the FSDP worker.
- Rollout engine: vLLM/HF/SGLang generation backend owned by the FSDP worker.
- Reference policy: either a separate FSDP worker group or a ref path inside the actor worker, depending on config.

The usual confusion is that `dp_actor.py` is not itself a Ray worker. It is the local PPO update implementation used inside each FSDP worker process.

## End-to-End Launch Path

For `eca_test.sh`, the call chain is:

1. `dapo_eca/eca_test.sh`
2. `ray job submit ... -- python3 -m dapo_eca.main_dapo ...`
3. `dapo_eca/main_dapo.py`
4. `TaskRunner.run(config)`
5. `RayDAPOTrainer.init_workers()`
6. `RayDAPOTrainer.fit()`

For many other scripts in the repo, the call chain is:

1. shell script
2. `python3 -m verl.trainer.main_ppo ...`
3. `verl/trainer/main_ppo.py`
4. `TaskRunner.run(config)`
5. `RayPPOTrainer.init_workers()`
6. `RayPPOTrainer.fit()`

The structure is the same. The main difference is that `dapo_eca.main_dapo` uses `RayDAPOTrainer`, which overrides parts of the PPO flow for DAPO-specific behavior and metrics.

## Step 1: What `eca_test.sh` Actually Does

`dapo_eca/eca_test.sh` does three important things:

1. Starts a local Ray head node.
2. Submits the repo as a Ray job working directory.
3. Launches `python3 -m dapo_eca.main_dapo` with Hydra overrides.

Important implications:

- The code used by workers is the packaged working directory Ray submits at launch time.
- A local code edit does nothing for an already running Ray job.
- The training script is not the long-lived controller process. The Ray job driver and then `TaskRunner` are.

## Step 2: `main_dapo.py` Creates the Driver Actor

`dapo_eca/main_dapo.py` does not run the whole training loop inline.

It:

- resolves the Hydra config,
- initializes Ray if needed,
- creates a remote `TaskRunner`,
- calls `ray.get(runner.run.remote(config))`.

So there is a small front-end process that submits work, and the real trainer logic runs inside the remote `TaskRunner`.

Why this exists:

- it keeps the control plane inside Ray,
- it makes resource ownership and profiling simpler,
- it matches the rest of the trainer stack.

## Step 3: `TaskRunner` Decides Which Worker Classes Exist

Inside `dapo_eca/main_dapo.py`, `TaskRunner.run()`:

- copies the model path locally,
- creates tokenizer and processor,
- chooses the worker classes from the strategy,
- builds a role-to-worker mapping,
- builds a `ResourcePoolManager`,
- instantiates `RayDAPOTrainer`,
- calls `trainer.init_workers()` and `trainer.fit()`.

For `eca_test.sh` specifically:

- actor strategy is `fsdp2`,
- critic strategy is `fsdp2`,
- rollout is vLLM,
- KL loss is enabled in config, so a reference policy role is created,
- reward model worker is not enabled, because reward is function-based in this script.

So the main roles are:

- `Role.ActorRollout`
- `Role.Critic`
- `Role.RefPolicy`

All three are mapped into the same global GPU pool.

## Step 4: Resource Pools and Worker Groups

`RayPPOTrainer.init_workers()` creates resource pools first.

Conceptually:

- a resource pool says how many worker processes should exist on each node,
- a worker group is the handle the trainer uses to call methods on all ranks of a role,
- placement groups pin those workers onto actual GPUs/nodes.

For `eca_test.sh` on one node with eight GPUs:

- one global pool is created,
- the pool describes eight GPU slots on one node,
- the trainer creates worker groups for actor/rollout, critic, and ref in that pool.

## Step 5: Why the Roles Look Separate but Share the Same Underlying Workers

This is the part that matters most when reading the code.

`RayPPOTrainer.init_workers()` groups all roles assigned to the same resource pool and passes them through `create_colocated_worker_cls(...)`.

That function builds one synthetic Ray class that contains multiple inner worker objects, one per role.

Then `RayWorkerGroup.spawn(...)` creates logical per-role views by rebinding prefixed methods.

So after `init_workers()`, the trainer has handles like:

- `self.actor_rollout_wg`
- `self.critic_wg`
- `self.ref_policy_wg`

but those can be different logical views over the same underlying colocated Ray actors on each GPU.

This is why the code feels like there are multiple independent worker groups even though the actual placement is more tightly fused.

## Step 6: What Lives Inside an FSDP Actor/Rollout/Ref Worker

The core FSDP class is `verl/workers/fsdp_workers.py::ActorRolloutRefWorker`.

Its role string determines what components it owns:

- actor
- rollout
- ref
- or a combination such as `actor_rollout_ref`

Important local objects created inside this worker:

- FSDP/FSDP2 wrapped model shards,
- optimizer and LR scheduler for the actor role,
- `DataParallelPPOActor` for actor updates or ref logprob computation,
- rollout engine for generation,
- checkpoint manager,
- Ulysses/FSDP sharding helpers.

For the actor role, `init_model()` creates:

- `self.actor_module_fsdp`
- `self.actor_optimizer`
- `self.actor_lr_scheduler`
- `self.actor = DataParallelPPOActor(...)`

For the rollout role, `init_model()` also builds the rollout backend.

For the ref role, it builds a separate ref model and wraps it in another `DataParallelPPOActor` instance used only for logprob evaluation.

## Step 7: What `DataParallelPPOActor` Actually Is

`verl/workers/actor/dp_actor.py::DataParallelPPOActor` is the local PPO implementation.

Despite the name, in your FSDP path it is not a Ray-distributed controller.

It is a per-worker helper that knows how to:

- run forward passes for logprobs and entropy,
- split mini-batches and micro-batches,
- compute PPO policy loss,
- compute KL loss,
- step the optimizer.

So the actual actor update chain is:

1. trainer calls `actor_rollout_wg.update_actor(batch)`
2. Ray dispatch reaches `AsyncActorRolloutRefWorker.update_actor(...)`
3. that calls `self.actor.update_policy(data)`
4. `self.actor` is a `DataParallelPPOActor`
5. `DataParallelPPOActor.update_policy(...)` runs the PPO loop locally on that worker rank

This is the cleanest mental model:

- Ray coordinates processes.
- FSDP worker owns model shards and mode switching.
- `DataParallelPPOActor` owns PPO math and optimizer stepping.

## Step 8: Training Step Flow in `RayDAPOTrainer.fit()`

The training loop itself runs on the driver inside `RayDAPOTrainer.fit()`.

The driver owns:

- dataloaders,
- prompt batching,
- reward combination logic,
- advantage computation,
- high-level sequencing of remote calls,
- logging and checkpoint policy.

One training step is roughly:

1. Read one prompt batch from the train dataloader.
2. Build a generation batch and repeat it `rollout.n` times.
3. Call rollout generation through `self.async_rollout_manager.generate_sequences(...)`.
4. Merge generated responses back into the batch.
5. Compute rewards.
6. Compute KL-related tensors if needed.
7. Compute critic values if critic is enabled.
8. Compute advantages on the driver.
9. Update critic via `self.critic_wg.update_critic(batch)`.
10. Update actor via `self.actor_rollout_wg.update_actor(batch)`.
11. Log metrics, validate occasionally, save checkpoints occasionally.

The heavy tensor work is mostly remote. The driver mostly orchestrates and performs light batch-level logic.

## Step 9: How Rollout Calls Work

The rollout path in your setup is async-hybrid.

The trainer does not call `actor_rollout_wg.generate_sequences(...)` directly in the DAPO path. It calls:

- `self.async_rollout_manager.generate_sequences(...)`

The async rollout manager sits on top of the actor/rollout worker group and manages:

- rollout server startup,
- request scheduling,
- switching between trainer mode and rollout mode,
- asynchronous generation behavior.

Inside the FSDP worker, generation typically looks like:

1. switch to rollout mode,
2. call the rollout backend,
3. switch back to trainer mode.

That mode switch is important in hybrid engine mode because the same worker can own both training-side model state and rollout-side serving state.

## Step 10: How Logprob and Ref Calls Work

There are three important remote calls besides rollout:

### `compute_log_prob`

Trainer call:

- `actor_rollout_wg.compute_log_prob(batch)`

Worker path:

- `ActorRolloutRefWorker.compute_log_prob(...)`
- then local `DataParallelPPOActor.compute_log_prob(...)`

Purpose:

- recompute `old_log_probs`,
- optionally entropy,
- optionally extra diagnostics like `sum_pi_squared`.

### `compute_ref_log_prob`

Trainer call:

- `ref_policy_wg.compute_ref_log_prob(batch)`

or, if ref is fused into actor:

- `actor_rollout_wg.compute_ref_log_prob(batch)`

Worker path:

- `ActorRolloutRefWorker.compute_ref_log_prob(...)`
- then local ref-side `DataParallelPPOActor.compute_log_prob(...)`

Purpose:

- compute `ref_log_prob` for KL penalty or KL loss.

### `update_actor`

Trainer call:

- `actor_rollout_wg.update_actor(batch)`

Worker path:

- `ActorRolloutRefWorker.update_actor(...)`
- then local `DataParallelPPOActor.update_policy(...)`

Purpose:

- run PPO update epochs over mini-batches and micro-batches,
- compute actor-side metrics,
- step the optimizer and LR scheduler.

## Step 11: How Critic Calls Work

Critic is a separate role and separate worker class.

Typical calls are:

- `critic_wg.compute_values(batch)`
- `critic_wg.update_critic(batch)`

The trainer uses critic values to compute advantages when the chosen advantage estimator requires a critic.

In GRPO-style setups the critic may be disabled, but the general worker wiring is the same.

## Step 12: DAPO-Specific Differences

`RayDAPOTrainer` overrides part of the generic PPO flow.

The main DAPO-specific hook is `compute_kl_related_metrics(...)`, which does things like:

- compute `response_mask`,
- compute `ref_log_prob`,
- recompute `old_log_probs`,
- attach rollout entropy,
- optionally compute ECA-related auxiliary tensors,
- optionally compute VN entropy.

This is why your DAPO path has more KL instrumentation than the generic `RayPPOTrainer` path.

The broader architecture is still the same:

- TaskRunner builds workers,
- trainer owns the loop,
- worker groups expose remote methods,
- FSDP worker owns local actor/rollout/ref objects.

## Step 13: Generic `main_ppo.py` vs `main_dapo.py`

The two entrypoints are structurally similar but differ in how much they specialize the trainer.

`verl.trainer.main_ppo`:

- is the generic entrypoint used by many example scripts,
- builds role mappings in a more configurable way,
- supports both legacy worker impl and new model-engine worker impl,
- instantiates `RayPPOTrainer`.

`dapo_eca.main_dapo`:

- is a narrower DAPO entrypoint,
- hardwires the DAPO trainer path,
- directly instantiates `RayDAPOTrainer`,
- uses DAPO-specific reward and KL instrumentation.

If you are reading code for your `eca_test.sh` run, prioritize `dapo_eca.main_dapo` and `dapo_eca.dapo_ray_trainer`.

## Step 14: Practical Mental Model

If you only remember one model, use this one:

- shell script packages config and code,
- Ray job launches a remote `TaskRunner`,
- `TaskRunner` builds a trainer and logical worker groups,
- each logical worker group is a handle over colocated Ray worker processes,
- each FSDP Ray worker owns model shards and local helper objects,
- `DataParallelPPOActor` is the local PPO math/optimizer object inside the worker,
- the driver sequences rollout -> reward -> KL/ref/old-logprob -> critic -> advantage -> actor update.

## Step 15: Concrete Call Graph for `eca_test.sh`

This is the simplest call graph for your current setup:

1. `eca_test.sh`
2. `ray job submit -- python3 -m dapo_eca.main_dapo ...`
3. `main_dapo.run_ppo(config)`
4. `TaskRunner.run(config)`
5. build `role_worker_mapping = {ActorRollout, Critic, RefPolicy}`
6. `trainer = RayDAPOTrainer(...)`
7. `trainer.init_workers()`
8. `actor_rollout_wg.init_model()`
9. `critic_wg.init_model()`
10. `ref_policy_wg.init_model()`
11. `async_rollout_manager = AgentLoopManager(...)`
12. `trainer.fit()`
13. per step:
14. dataloader batch on driver
15. `async_rollout_manager.generate_sequences(...)`
16. reward computation
17. `ref_policy_wg.compute_ref_log_prob(...)`
18. `actor_rollout_wg.compute_log_prob(...)`
19. `critic_wg.compute_values(...)`
20. advantage computation on driver
21. `critic_wg.update_critic(...)`
22. `actor_rollout_wg.update_actor(...)`
23. logging/checkpoint/validation

## Common Misunderstandings

### "Is `dp_actor.py` the distributed trainer?"

No.

It is the local PPO actor implementation used inside each FSDP worker rank.

### "Is the trainer itself distributed?"

The trainer control loop is centralized on the driver.

The heavy compute is distributed through worker groups.

### "Does one Ray worker equal one model?"

Not necessarily.

In the colocated path, one underlying Ray worker process can host multiple logical role objects.

### "Does `actor_rollout_wg` mean pure actor only?"

No.

In your path it refers to the worker group that owns the training actor and rollout functionality, and sometimes ref functionality depending on config.

## Recommended Reading Order

If you want to understand the code quickly, read in this order:

1. `dapo_eca/eca_test.sh`
2. `dapo_eca/main_dapo.py`
3. `dapo_eca/dapo_ray_trainer.py`
4. `verl/trainer/ppo/ray_trainer.py`
5. `verl/single_controller/ray/base.py`
6. `verl/workers/fsdp_workers.py`
7. `verl/workers/actor/dp_actor.py`

That order goes from launch, to trainer orchestration, to Ray dispatch, to worker internals, to PPO math.
