import torch

def normalize_data(data, min_vals, max_vals):
    min_vals = min_vals.unsqueeze(0).unsqueeze(0)  
    max_vals = max_vals.unsqueeze(0).unsqueeze(0)  

    normalized_data = (data - min_vals) / (max_vals - min_vals)

    return normalized_data

if __name__ == "__main__":
    bs = 32
    nf = 10
    data = torch.randn((bs, nf, 6))  

    # means = torch.tensor([2.17239228e-02 -8.76334959e-01 1.83403242e-01 4.68812609e-04 6.09114990e+01 6.82846017e+01])
    # stds = torch.tensor([3.95977561e+00 2.74141379e+00 2.70259097e+00 8.42982963e-06 1.71036724e+00 1.94872744e+00])
    min_vals = torch.tensor([-1.03461033e+01, -8.08477430e+00, -7.56659334e+00, 4.33026857e-04, 5.68175623e+01, 6.36141304e+01])
    max_vals = torch.tensor([1.75214498e+01, 8.44862517e+00, 7.98321722e+00, 6.12732050e-04, 6.88481830e+01, 8.21925801e+01])

    normalized_data = normalize_data(data, min_vals, max_vals)
    print(normalized_data)
