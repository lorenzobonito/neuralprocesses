import lab as B
import torch
import matplotlib.pyplot as plt


dataset = torch.load("/scratch/lb953/benchmark_datasets/benchmark_dataset_noised_gp_1_layers.pt", map_location="cuda")

# Evaluate model predictions over context sets
json_data = {}
for idx, batch in enumerate(dataset):    
    cont_size = batch["contexts"][0][0].numel()
    if cont_size in json_data:
        json_data[cont_size] = torch.concat((json_data[cont_size], batch["pred_logpdf_diag"]/batch["xt"][0][0].numel()), dim=0)
    else:
        json_data[cont_size] = batch["pred_logpdf_diag"]/batch["xt"][0][0].numel()

for cont_size in json_data.keys():
    json_data[cont_size] = torch.exp(json_data[cont_size].mean())
values = [value.item() for value in json_data.values()]
plt.plot(values[:19])
plt.savefig("test.png")
#     json_data[batch["contexts"][0][0].numel()] = (batch["contexts"][0][0].numel(), loglik.item())
#     out.kv(f"Dataset {idx}", (str(batch["contexts"][0][0].numel()), *loglik))
#     with open(wd_eval.file("logliks.json"), "w", encoding="utf-8") as f:
#         json.dump(json_data, f, ensure_ascii=False, indent=4)

# logliks = B.concat(*logliks)
# out.kv("Loglik (P)", exp.with_err(logliks, and_lower=True))
