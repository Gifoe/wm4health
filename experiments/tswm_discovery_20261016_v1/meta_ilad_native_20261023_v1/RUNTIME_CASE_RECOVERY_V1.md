# Preserve environment-provisioning failure; resume the verified empty venv

The first runtime launcher created the isolated Python3.11 venv successfully, then our setup code raised KeyError('SystemRoot') before any package installation. Python's os.environ is case-insensitive on Windows, but its plain-dict copy contains uppercase keys. This was our environment-wrapper mistake, not an upstream package or model failure.

Recovery checks the prior venv-creation exit0, the exact planned G runtime path, the preserved KeyError and the existing Python3.11 executable. It reuses that just-created environment without deleting or recreating it, resolves SYSTEMROOT with a case-aware fallback, and then performs the unchanged package-installation and qualification steps. A fresh recovery output/exit receipt preserves the first failure. No model execution, scientific fit or shared-environment modification occurred in the failed attempt.
