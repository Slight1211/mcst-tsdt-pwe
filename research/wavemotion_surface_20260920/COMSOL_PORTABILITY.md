# COMSOL controller prerequisites

The public release contains COMSOL Java source and archived Python controllers.
It does not contain COMSOL, compiled `.class` builders, private `.prefs` files,
historical `.mph` checkpoints, or most historical run directories.

Set `COMSOL_BIN` to your local installation's bin directory containing
`comsolbatch.exe`. Set `COMSOL_PREFS_FILE` to a local batch preference file for
the fresh/root controllers. The archived joint controllers retain their original
preference-directory prerequisites and frozen input hashes; supplying unrelated
preferences or checkpoints does not satisfy those checks. Compile each required
Java source with your licensed COMSOL installation before using its controller.

Path checks run when COMSOL is requested. Importing PWE modules does not require
COMSOL configuration. Missing private inputs produce an explicit error; this
release does not recreate them or bypass an input hash/source-difference check.

The Java builder accepts `COMSOL_MODEL_DIR`, `COMSOL_RESULT_CSV`, and
`COMSOL_TIMING_CSV`; its fallback paths are relative to the batch working directory.
`ExportSection41Meshes` accepts `MCST_PROJECT_ROOT`, `COMSOL_MESH_RUN`, and
`COMSOL_MESH_ASSETS`. Saved-model tools require `MCST_JOINT_INPUT` or
`MCST_FIELD_MODEL`, as named in each source file.

Joint controllers retain Windows PowerShell preflight and their original
historical prerequisites, solver parameters, hashes, and completion requirements.
The thickness queue's local process identifier is configured with
`SCALE_CONTROLLER_PID`; restore the referenced scale run first. These controllers
remain archival workflows rather than standalone examples. Original batches can
run without a time limit; select and inspect an entry point before execution.

No COMSOL or numerical validation was run while preparing this public copy.
