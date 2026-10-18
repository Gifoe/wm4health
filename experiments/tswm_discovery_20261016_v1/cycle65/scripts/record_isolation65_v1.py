from common65 import *
import sys,site,numpy,torch,scipy,importlib.util
assert not site.ENABLE_USER_SITE and sys.flags.isolated
dump(OUT/'RUNTIME_ISOLATION65_V1.json',dict(utc=utc(),status='ISOLATED_NATIVE_FIXTURE_PASSED',python=sys.version,executable=sys.executable,prefix=sys.prefix,user_site_enabled=site.ENABLE_USER_SITE,numpy=numpy.__version__,numpy_origin=numpy.__file__,torch=torch.__version__,torch_origin=torch.__file__,scipy=scipy.__version__,inactive_optional_import_omitted=importlib.util.find_spec('linear_attention_transformer') is None,modified_native_algorithm_bodies=0,original_qualification_sha256=sha(OUT/'QUALIFICATION65_V1.json'),isolated_qualification_sha256=sha(OUT/'QUALIFICATION_ISOLATED65_V1.json'),source_receipt_sha256=sha(OUT/'SOURCES65_V1.json'),root_cause_of_native_failures='Not established; user-site version difference verified'))
print(json.dumps(dict(status='ISOLATED_NATIVE_FIXTURE_PASSED',numpy=numpy.__version__,torch=torch.__version__)))
