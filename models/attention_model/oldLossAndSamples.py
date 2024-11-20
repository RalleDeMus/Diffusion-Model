
# # Loss calculation function
# def calc_loss(score_network: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
#     t = torch.rand((x.shape[0], 1), dtype=x.dtype, device=x.device) * (1 - 1e-4) + 1e-4
#     int_beta = (0.1 + 0.5 * (20 - 0.1) * t) * t
#     int_beta = int_beta.view(-1, 1, 1, 1).expand_as(x)
#     mu_t = x * torch.exp(-0.5 * int_beta)
#     var_t = -torch.expm1(-int_beta)
#     x_t = torch.randn_like(x) * var_t ** 0.5 + mu_t
#     grad_log_p = -(x_t - mu_t) / var_t
#     score = score_network(x_t, t)
#     loss = (score - grad_log_p) ** 2
#     lmbda_t = var_t
#     weighted_loss = lmbda_t * loss
#     return torch.mean(weighted_loss)

# # Define sample generation function
# def generate_samples(score_network: torch.nn.Module, nsamples: int, image_shape) -> torch.Tensor:
#     device = next(score_network.parameters()).device
#     x_t = torch.randn((nsamples, *image_shape), device=device)
#     time_pts = torch.linspace(1, 0, 1000, device=device)
#     beta = lambda t: 0.1 + (20 - 0.1) * t
#     for i in range(len(time_pts) - 1):
#         t = time_pts[i]
#         dt = time_pts[i + 1] - t
#         fxt = -0.5 * beta(t) * x_t
#         gt = beta(t) ** 0.5
#         score = score_network(x_t, t.expand(x_t.shape[0], 1)).detach()
#         drift = fxt - gt * gt * score
#         diffusion = gt
#         x_t = x_t + drift * dt + diffusion * torch.randn_like(x_t) * torch.abs(dt) ** 0.5
#     return x_t