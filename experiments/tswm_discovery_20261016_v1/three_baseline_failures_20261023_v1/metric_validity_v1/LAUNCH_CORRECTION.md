# Before-launch capture correction

The unexecuted v1 capture expected the scientific output directory to exist, while the worker intentionally creates that directory exclusively. Version 2 writes its log and process receipts in the existing parent campaign outputs, leaving the new scientific outputs to the original worker. The scientific protocol and worker are unchanged. Version 1 remains as the source of the integration error; it was never launched and produced no scientific result. A launch/transport issue is not model failure.
