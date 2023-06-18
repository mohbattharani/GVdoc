import sys, os
sys.path.append("..")

from evaluation import SCORES, compute_plot_cm
scores = SCORES()


f1 = "logs/rvlcdip_n_GraphLayoutv6_5.json"
f2 = "logs/rvlcdip_n_GraphLayoutv6_10.json"
f3 = "logs/rvlcdip_n_GraphLayoutv6_15.json"
f4 = "logs/rvlcdip_n_GraphLayoutv6_20.json"
f5 = "logs/rvlcdip_n_GraphLayoutv6_30.json"

Fs = [f1, f2, f3, f4, f5]
for f in Fs:
    scores(f, do_each=False)
    

f1 = "logs/rvlcdip_GraphLayoutv6_5.json"
f2 = "logs/rvlcdip_GraphLayoutv6_10.json"
f3 = "logs/rvlcdip_GraphLayoutv6_15.json"
f4 = "logs/rvlcdip_GraphLayoutv6_20.json"
f5 = "logs/rvlcdip_GraphLayoutv6_30.json"

Fs = [f1, f2, f3, f4]
for f in Fs:
    scores(f, do_each=False)
    

scores(f1, do_each=False)

print (" ================  Jan 14  =====================")
f1 = "logs/Jan14_rvlcdip_n_GraphLayoutv6_10.json"
f2 = "logs/Jan14_rvlcdip_n_GraphLayoutv6_20.json"
f3 = "logs/Jan14_rvlcdip_n_GraphLayoutv6_30.json"


scores(f1, do_each=False)
scores(f2, do_each=False)
scores(f3, do_each=False)


print (" =============== Jan 13  ======================")
f1 = "logs/Jan13_rvlcdip_n_GraphLayoutv6_10.json"
f2 = "logs/Jan13_rvlcdip_n_GraphLayoutv6_25.json"
f3 = "logs/Jan13_rvlcdip_GraphLayoutv6_10.json"
f4 = "logs/Jan13_rvlcdip_GraphLayoutv6_25.json"

scores(f1, do_each=False)
scores(f2, do_each=False)
scores(f3, do_each=False)
scores(f4, do_each=False)