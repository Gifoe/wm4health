import runtime_profile
from data_protocol import prepare,references,Data
def main():
 prepare();d=Data();d.batch(d.train[:16],'train');d.batch(d.val[:16],'val');references()
if __name__=='__main__':main()
