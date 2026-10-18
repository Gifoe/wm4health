from common87_v1 import *
def main():
 receipt=read(O/'NUMERICAL_QUALIFICATION87_V1.json');assert not receipt['full_population_continuous_time_error_certified']
 old=(C/'POSTSTUDY_ANALYSIS87_V1.md').read_text(encoding='utf-8');prefix='# Numerical qualification update for the paired analysis\n\nThese are descriptive comparisons under the fixed RK4/TF32 protocol. Supplementary independent learned-rollout checks did not pass1% on the frozen retained-cohort example; adapted-model checks were not reached. Do not interpret54/54 as a certified continuous-time comparison or a general theorem. See NUMERICAL_OUTCOME87_V1.md. The scoped method route is deprioritized for lack of sufficient novelty/necessity evidence; every possible trajectory-selection effect is not falsified.\n\n'
 with (C/'POSTSTUDY_ANALYSIS87_V2.md').open('x',encoding='utf-8') as f:f.write(prefix+old)
if __name__=='__main__':main()
