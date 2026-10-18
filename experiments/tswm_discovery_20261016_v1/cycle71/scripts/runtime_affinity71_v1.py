"""Query Windows CPU sets; optionally restrict only the current process."""
import ctypes, struct, collections
from ctypes import wintypes

def configure(mode='read'):
    assert mode in ('read','unrestricted','efficient')
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.GetCurrentProcess.restype=ctypes.c_void_p
    current=kernel.GetCurrentProcess()
    kernel.GetSystemCpuSetInformation.argtypes=[ctypes.c_void_p,wintypes.ULONG,ctypes.POINTER(wintypes.ULONG),ctypes.c_void_p,wintypes.ULONG]
    kernel.GetSystemCpuSetInformation.restype=wintypes.BOOL
    needed=wintypes.ULONG()
    kernel.GetSystemCpuSetInformation(None,0,ctypes.byref(needed),current,0)
    assert needed.value>0
    buffer=ctypes.create_string_buffer(needed.value)
    assert kernel.GetSystemCpuSetInformation(buffer,needed.value,ctypes.byref(needed),current,0),ctypes.get_last_error()
    raw=buffer.raw[:needed.value];position=0;rows=[]
    while position<len(raw):
        size,kind=struct.unpack_from('<II',raw,position)
        assert size>=8 and position+size<=len(raw)
        if kind==0:
            assert size>=32
            ident,group,logical,core,cache,numa,efficiency,flags=struct.unpack_from('<IHBBBBBB',raw,position+8)
            rows.append(dict(id=ident,group=group,logical_processor=logical,core=core,efficiency_class=efficiency,flags=flags))
        position+=size
    assert position==len(raw)
    kernel.GetProcessAffinityMask.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_size_t),ctypes.POINTER(ctypes.c_size_t)]
    kernel.GetProcessAffinityMask.restype=wintypes.BOOL
    kernel.SetProcessAffinityMask.argtypes=[ctypes.c_void_p,ctypes.c_size_t]
    kernel.SetProcessAffinityMask.restype=wintypes.BOOL
    original=ctypes.c_size_t();system=ctypes.c_size_t()
    assert kernel.GetProcessAffinityMask(current,ctypes.byref(original),ctypes.byref(system))
    eligible=[r for r in rows if r['group']==0 and original.value&(1<<r['logical_processor']) and not (r['flags']&2 and not r['flags']&4)]
    classes=sorted({r['efficiency_class'] for r in eligible})
    chosen=[]
    if mode=='efficient':
        assert len(classes)>=2,'No verified heterogeneous CPU classes; do not guess a CPU mask'
        chosen=[r for r in eligible if r['efficiency_class']==classes[0]]
        assert len(chosen)>=4
        mask=sum(1<<r['logical_processor'] for r in chosen)
        assert kernel.SetProcessAffinityMask(current,mask),ctypes.get_last_error()
    after=ctypes.c_size_t();system_after=ctypes.c_size_t()
    assert kernel.GetProcessAffinityMask(current,ctypes.byref(after),ctypes.byref(system_after))
    if mode=='efficient':assert after.value==mask
    else:assert after.value==original.value
    return dict(mode=mode,original_process_mask=hex(original.value),current_process_mask=hex(after.value),
                available_efficiency_classes=classes,selected_logical_processors=[r['logical_processor'] for r in chosen],
                cpu_sets=rows,system_configuration_modified=False,other_processes_modified=False,
                interpretation='Process-local execution control, not a diagnosis or a proven hardware remedy')

if __name__=='__main__':
    import json
    print(json.dumps(configure('read')),flush=True)
