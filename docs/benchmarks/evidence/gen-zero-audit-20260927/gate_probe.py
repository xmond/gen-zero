from gen_zero.gate.policy_gate import DecisionPolicyGate,DomainRiskProfile
for risk in [1.0,float('nan')]:
 p=DomainRiskProfile(whitelisted_targets={'fixture-target'})
 d={'target':'fixture-target','action':'delete','confidence':0.99,'risk':risk}
 print('input',repr(d)); print('verdict',DecisionPolicyGate(p).evaluate_policy(d).to_dict())
print('missing',DecisionPolicyGate().evaluate_policy({}).to_dict())
