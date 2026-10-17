# DCL public data: server access confirmed

At 2026-10-17T19:19:14.400668+00:00, the public record API returned HTTP200 with 22 file entries and 4,964,072,376 advertised bytes in total. These are catalog sizes, not downloaded signal bytes. No signals were downloaded. The exact filenames, provider MD5 checksums, content links and reported rights are retained in outputs/dcl_catalog_summary54_v1.json.

The first Python urllib metadata probe failed to resolve the certificate issuer. A separate request through unchanged native Windows certificate validation succeeded for the record page and API. This establishes metadata availability from this server; it does not establish that every signal file has been downloaded, hash-verified or decoded. Both transport receipts are preserved. No insecure TLS fallback was used.

The source/input boundary remains DCL_INFERENCE_CONTRACT54_V1.md. Future use should select only files that answer a discriminating mechanism question, freeze their advertised and actual hashes, and disclose the training/heldout protocol. Do not run the full Table1 sweep just to increase reproduction count. No DCL fit or forecast comparison is claimed here.

Catalog summary SHA256: 6ee509bc3dd28b566ae2d039a902a285416dec15aa70aada789741ccd8bb2240
Raw native TLS receipt SHA256: 4bfd98ea7166b99a40cc5be8f3d85df24d36a4d7e1c0dcaa0b04c6f2e845b3cd
