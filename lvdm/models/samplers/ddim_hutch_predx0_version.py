import numpy as np
from tqdm import tqdm
import torch
from lvdm.models.utils_diffusion import make_ddim_sampling_parameters, make_ddim_timesteps, rescale_noise_cfg
from lvdm.common import noise_like
from lvdm.common import extract_into_tensor

# --- JEPA guidance (Hutchinson energy) ---
from contextlib import nullcontext
# from energy.jepa_score import hutchinson_trace_jtj
from energy.jepa_score import fd_hutchinson_trace_jtj


class DDIMSampler(object):
    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule
        self.counter = 0

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps, verbose=verbose)
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        if self.model.use_dynamic_rescale:
            self.ddim_scale_arr = self.model.scale_arr[self.ddim_timesteps]
            self.ddim_scale_arr_prev = torch.cat([self.ddim_scale_arr[0:1], self.ddim_scale_arr[:-1]])

        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta, verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                    1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    @torch.no_grad()
    def sample(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               schedule_verbose=False,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               precision=None,
               fs=None,
               timestep_spacing='uniform',  # uniform_trailing for starting from last timestep
               guidance_rescale=0.0,
               **kwargs
               ):

        # check condition bs
        if conditioning is not None:
            if isinstance(conditioning, dict):
                try:
                    cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                except:
                    cbs = conditioning[list(conditioning.keys())[0]][0].shape[0]

                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_discretize=timestep_spacing, ddim_eta=eta, verbose=schedule_verbose)

        # make shape
        if len(shape) == 3:
            C, H, W = shape
            size = (batch_size, C, H, W)
        elif len(shape) == 4:
            C, T, H, W = shape
            size = (batch_size, C, T, H, W)

        samples, intermediates = self.ddim_sampling(conditioning, size,
                                                    callback=callback,
                                                    img_callback=img_callback,
                                                    quantize_denoised=quantize_x0,
                                                    mask=mask, x0=x0,
                                                    ddim_use_original_steps=False,
                                                    noise_dropout=noise_dropout,
                                                    temperature=temperature,
                                                    score_corrector=score_corrector,
                                                    corrector_kwargs=corrector_kwargs,
                                                    x_T=x_T,
                                                    log_every_t=log_every_t,
                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                    unconditional_conditioning=unconditional_conditioning,
                                                    verbose=verbose,
                                                    precision=precision,
                                                    fs=fs,
                                                    guidance_rescale=guidance_rescale,
                                                    **kwargs)
        return samples, intermediates

    @torch.no_grad()
    def ddim_sampling(self, cond, shape,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None, verbose=True, precision=None,
                      fs=None, guidance_rescale=0.0,
                      **kwargs):
        device = self.model.betas.device
        b = shape[0]

        if x_T is None:
            img = torch.randn(shape, device=device)  # x_T가 주어지지 않으면 → 랜덤 노이즈에서 시작
        else:
            img = x_T  # x_T가 주어지면 → 그 노이즈를 그대로 시작점으로 사용

        # DynamiCrafter는 첫 프레임 이미지는 전혀 x_T에 들어가지 않는다.
        # 조건(cond)으로만 사용되고 sampling 시작점은 항상 pure noise
        # DynamiCrafter는 첫번째 프레임을 Denoising UNet의 cross-attention layer에 추가한다.
        # TI2V-Zero나 T2V-Zero와 헷갈리지 말자
        # DynamiCrafter의 장점은 pure noise에서 시작하기 때문에, 더 다양한 동영상을 생성할 수 있다.

        if precision is not None:
            if precision == 16:
                img = img.to(dtype=torch.float16)

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        # DDIM 샘플링 중간 결과를 저장하기 위한 로그 컨테이너
        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        # JEPA anomaly score logs (filled only if enabled)
        self._jepa_anomaly_fd_log = []

        time_range = reversed(range(0, timesteps)) if ddim_use_original_steps else np.flip(timesteps)

        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        if verbose:
            iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)
        else:
            iterator = time_range

        clean_cond = kwargs.pop("clean_cond", False)

        # cond_copy, unconditional_conditioning_copy = copy.deepcopy(cond), copy.deepcopy(unconditional_conditioning)
        for i, step in enumerate(iterator):

            # DDIM 디노이징 단계(step index)
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            ## use mask to blend noised original latent (img_orig) & new sampled latent (img)
            if mask is not None:
                assert x0 is not None
                if clean_cond:
                    img_orig = x0
                else:
                    img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass? <ddim inversion>

                img = img_orig * mask + (1. - mask) * img  # keep original & modify use img

            outs = self.p_sample_ddim(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                      quantize_denoised=quantize_denoised, temperature=temperature,
                                      noise_dropout=noise_dropout, score_corrector=score_corrector,
                                      corrector_kwargs=corrector_kwargs,
                                      unconditional_guidance_scale=unconditional_guidance_scale,
                                      unconditional_conditioning=unconditional_conditioning,
                                      mask=mask, x0=x0, fs=fs, guidance_rescale=guidance_rescale,
                                      **kwargs)

            # JEPA-SCORE energy는 x_t에 걸어야 할까, pred_x0에 걸어야 할까?

            img, pred_x0 = outs  # x_prev, pred_x0

            if callback: callback(i)
            if img_callback: img_callback(pred_x0, i)

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)  # denosing step 별 latent

        # attach anomaly logs if any
        if len(getattr(self, '_jepa_anomaly_fd_log', [])) > 0:
            intermediates['jepa_anomaly_fd'] = self._jepa_anomaly_fd_log

        return img, intermediates

    @torch.no_grad()
    def p_sample_ddim(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,
                      uc_type=None, conditional_guidance_scale_temporal=None, mask=None, x0=None, guidance_rescale=0.0,
                      **kwargs):

        # x: 현재 latent
        # c: conditioning (텍스트/이미지 조건 등)
        # t: 현재 timestep 텐서(배치 크기만큼 동일 값)
        # index: DDIM timestep 배열에서의 인덱스

        b, *_, device = *x.shape, x.device

        if x.dim() == 5:
            is_video = True
        else:
            is_video = False

        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            model_output = self.model.apply_model(x, t, c, **kwargs)  # unet denoiser
        else:
            ### do_classifier_free_guidance
            if isinstance(c, torch.Tensor) or isinstance(c, dict):
                e_t_cond = self.model.apply_model(x, t, c, **kwargs)
                e_t_uncond = self.model.apply_model(x, t, unconditional_conditioning, **kwargs)
            else:
                raise NotImplementedError

            model_output = e_t_uncond + unconditional_guidance_scale * (e_t_cond - e_t_uncond)

            if guidance_rescale > 0.0:
                model_output = rescale_noise_cfg(model_output, e_t_cond, guidance_rescale=guidance_rescale)

        if self.model.parameterization == "v":  # ????????????
            e_t = self.model.predict_eps_from_z_and_v(x, t, model_output)
        else:
            e_t = model_output

        if score_corrector is not None:
            assert self.model.parameterization == "eps", 'not implemented'
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        # sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        sigmas = self.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep

        if is_video:
            size = (b, 1, 1, 1, 1)
        else:
            size = (b, 1, 1, 1)
        a_t = torch.full(size, alphas[index], device=device)
        a_prev = torch.full(size, alphas_prev[index], device=device)
        sigma_t = torch.full(size, sigmas[index], device=device)
        sqrt_one_minus_at = torch.full(size, sqrt_one_minus_alphas[index], device=device)

        # current prediction for x_0
        if self.model.parameterization != "v":
            pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()  # 현재 denosing step에서 예측 값
        else:
            pred_x0 = self.model.predict_start_from_z_and_v(x, t, model_output)

        if self.model.use_dynamic_rescale:
            scale_t = torch.full(size, self.ddim_scale_arr[index], device=device)
            prev_scale_t = torch.full(size, self.ddim_scale_arr_prev[index], device=device)
            rescale = (prev_scale_t / scale_t)
            pred_x0 *= rescale

        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)

        # -----------------------
        # JEPA Forward-Difference Guidance (TITAN-Guide style)
        # -----------------------
        jepa_cfg = kwargs.get("jepa_cfg", None)
        if jepa_cfg is not None:
            cfg = self._default_jepa_cfg()
            cfg.update(jepa_cfg)
            jepa_cfg = cfg
        else:
            # 아예 jepa_cfg를 안 넘겼을 때
            jepa_cfg = None

        if jepa_cfg is not None and jepa_cfg.get("enable", False):
            assert "encoder_fn" in jepa_cfg and jepa_cfg["encoder_fn"] is not None, \
                "jepa_cfg['encoder_fn'] must be provided (e.g., V-JEPA2 encoder callable)."

            # progress: 0 (early) -> 1 (late)
            total_steps = len(self.ddim_timesteps) if not use_original_steps else self.ddpm_num_timesteps
            progress = 1.0 - (index / max(total_steps - 1, 1))

            t0 = float(jepa_cfg.get("t_start_ratio", 0.0))
            t1 = float(jepa_cfg.get("t_end_ratio", 1.0))
            in_window = (progress >= t0) and (progress <= t1)

            every_k = int(jepa_cfg.get("every_k", 1))
            do_step = (every_k <= 1) or ((index % every_k) == 0)

            if in_window and do_step:

                print(
                    f"[JEPA GUIDE CHECK] index={index}, progress={progress:.3f}, "
                    f"in_window={in_window}, do_step={do_step}"
                )

                # 방향 V 선택: score(e_t) 또는 random
                v_mode = jepa_cfg.get("v_mode", "score")
                if v_mode == "score":
                    V = e_t.detach()
                else:
                    V = torch.randn_like(pred_x0)

                # per-sample normalize (안 하면 스케일에 종속됨)
                if jepa_cfg.get("normalize_v", True):
                    v_flat = V.reshape(V.shape[0], -1)
                    v_norm = v_flat.norm(dim=1).clamp_min(1e-8).reshape(
                        -1, *([1] * (V.dim() - 1))
                    )
                    V = V / v_norm

                fd_eps = float(jepa_cfg.get("fd_eps", 1e-3))
                maximize = bool(jepa_cfg.get("maximize", True))

                # lambda schedule inside window
                frac = (progress - t0) / max(t1 - t0, 1e-8)  # 0..1
                lam_t = self._jepa_lambda_t(jepa_cfg, frac)

                # fp32로 energy 계산(권장)
                use_fp32 = bool(jepa_cfg.get("use_fp32", True))
                amp_ctx = torch.cuda.amp.autocast(enabled=False) if use_fp32 else nullcontext()

                with amp_ctx:

                    # Fix Hutchinson probes across E0/E1 within this denoising step
                    base_seed = int(jepa_cfg.get("hutch_seed", 1234))
                    step_seed = base_seed + int(index)
                    torch.manual_seed(step_seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(step_seed)
                    E0 = self._compute_jepa_energy(pred_x0, jepa_cfg)  # (B,)
                    pred_pert = pred_x0 + fd_eps * V

                    # Fix Hutchinson probes across E0/E1 within this denoising step
                    base_seed = int(jepa_cfg.get("hutch_seed", 1234))
                    step_seed = base_seed + int(index)
                    torch.manual_seed(step_seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(step_seed)
                    E1 = self._compute_jepa_energy(pred_pert, jepa_cfg)  # (B,)

                    # maximize energy: loss = -E, minimize: loss = +E
                    L0 = -E0 if maximize else E0
                    L1 = -E1 if maximize else E1

                    print(
                        "E0 mean/std:", E0.mean().item(), E0.std().item(),
                        "E1 mean/std:", E1.mean().item(), E1.std().item(),
                        "deltaE:", (E1 - E0).abs().mean().item()
                    )

                    # directional grad estimate along V:
                    # g ≈ ((L1 - L0)/eps) * V   (broadcast per-sample scalar)
                    delta = (L1 - L0) / max(fd_eps, 1e-12)  # (B,)
                    delta = delta.reshape(-1, *([1] * (V.dim() - 1)))
                    g = delta * V

                # pred_x0 수정
                # pred_x0_old = pred_x0.detach().clone()

                pred_x0 = (pred_x0 - lam_t * g).type_as(pred_x0)

                # pred_x0_new = pred_x0.detach()  # 또는 clone()

                # dx = (pred_x0_new - pred_x0_old).float()
                # print("||Δpred_x0||_mean", dx.abs().mean().item(),
                #       "||Δpred_x0||_max", dx.abs().max().item(),
                #       flush=True)

        # -----------------------
        # JEPA Forward-Difference Anomaly Score (no update)
        # -----------------------
        if jepa_cfg is not None and jepa_cfg.get("anomaly_fd", False):
            assert "encoder_fn" in jepa_cfg and jepa_cfg["encoder_fn"] is not None, \
                "jepa_cfg['encoder_fn'] must be provided (e.g., V-JEPA2 encoder callable)."

            total_steps = len(self.ddim_timesteps) if not use_original_steps else self.ddpm_num_timesteps
            progress = 1.0 - (index / max(total_steps - 1, 1))

            t0 = float(jepa_cfg.get("t_start_ratio", 0.0))
            t1 = float(jepa_cfg.get("t_end_ratio", 1.0))
            in_window = (progress >= t0) and (progress <= t1)

            every_k = int(jepa_cfg.get("every_k", 1))
            do_step = (every_k <= 1) or ((index % every_k) == 0)

            anomaly_val = None
            if in_window and do_step:
                # how many probe directions (RMS)
                # (backward compat: allow n_dir to specify this as well)
                K = int(jepa_cfg.get("anomaly_n_dir", jepa_cfg.get("n_dir", 1)))
                fd_eps = float(jepa_cfg.get("fd_eps", 1e-3))

                # baseline energy (B,)
                use_fp32 = bool(jepa_cfg.get("use_fp32", True))
                amp_ctx = torch.cuda.amp.autocast(enabled=False) if use_fp32 else nullcontext()

                with amp_ctx:

                    # Fix Hutchinson probes across E0/E1 within this denoising step
                    base_seed = int(jepa_cfg.get("hutch_seed", 1234))
                    step_seed = base_seed + int(index)
                    torch.manual_seed(step_seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(step_seed)
                    E0 = self._compute_jepa_energy(pred_x0, jepa_cfg)  # (B,)

                    deltas = []
                    v_mode = jepa_cfg.get("v_mode", "score")
                    for _ in range(max(K, 1)):
                        if v_mode == "score":
                            V = e_t.detach()
                        else:
                            V = torch.randn_like(pred_x0)

                        if jepa_cfg.get("normalize_v", True):
                            v_flat = V.reshape(V.shape[0], -1)
                            v_norm = v_flat.norm(dim=1).clamp_min(1e-8).reshape(
                                -1, *([1] * (V.dim() - 1))
                            )
                            V = V / v_norm

                        pred_pert = pred_x0 + fd_eps * V

                        # Fix Hutchinson probes across E0/E1 within this denoising step
                        base_seed = int(jepa_cfg.get("hutch_seed", 1234))
                        step_seed = base_seed + int(index)
                        torch.manual_seed(step_seed)
                        if torch.cuda.is_available():
                            torch.cuda.manual_seed_all(step_seed)
                        E1 = self._compute_jepa_energy(pred_pert, jepa_cfg)  # (B,)
                        deltas.append((E1 - E0) / max(fd_eps, 1e-12))  # (B,)

                    # RMS over directions -> (B,)
                    delta = torch.stack(deltas, dim=0)  # (K,B)
                    anomaly_val = torch.sqrt((delta ** 2).mean(dim=0))

            # log
            try:
                if getattr(self, "_jepa_anomaly_fd_log", None) is not None:
                    self._jepa_anomaly_fd_log.append({
                        "index": int(index),
                        "t": int(t[0].item()) if isinstance(t, torch.Tensor) else int(t),
                        "progress": float(progress),
                        "in_window": bool(in_window),
                        "do_step": bool(do_step),
                        "anomaly": None if anomaly_val is None else anomaly_val.detach().float().cpu(),
                    })
            except Exception:
                pass

        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t ** 2).sqrt() * e_t

        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)

        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        # x_prev = (clean 성분) + (방향 성분) + (랜덤성)

        # TITAN-Guide류가 주로 x_prev가 아닌 pred_x0를 수정한다
        # pred_x0는 "모델이 생각하는 clean"이기 때문에

        '''
        x_t
        ↓  (denoiser)
        pred_x0
        ↓  (energy / guidance)
        x_prev
        '''

        return x_prev, pred_x0

    @torch.no_grad()
    def decode(self, x_latent, cond, t_start, unconditional_guidance_scale=1.0, unconditional_conditioning=None,
               use_original_steps=False, callback=None):

        timesteps = np.arange(self.ddpm_num_timesteps) if use_original_steps else self.ddim_timesteps
        timesteps = timesteps[:t_start]

        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='Decoding image', total=total_steps)
        x_dec = x_latent
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((x_latent.shape[0],), step, device=x_latent.device, dtype=torch.long)
            x_dec, _ = self.p_sample_ddim(x_dec, cond, ts, index=index, use_original_steps=use_original_steps,
                                          unconditional_guidance_scale=unconditional_guidance_scale,
                                          unconditional_conditioning=unconditional_conditioning)
            if callback: callback(i)
        return x_dec

    @torch.no_grad()
    def stochastic_encode(self, x0, t, use_original_steps=False, noise=None):
        # fast, but does not allow for exact reconstruction
        # t serves as an index to gather the correct alphas
        if use_original_steps:
            sqrt_alphas_cumprod = self.sqrt_alphas_cumprod
            sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod
        else:
            sqrt_alphas_cumprod = torch.sqrt(self.ddim_alphas)
            sqrt_one_minus_alphas_cumprod = self.ddim_sqrt_one_minus_alphas

        if noise is None:
            noise = torch.randn_like(x0)
        return (extract_into_tensor(sqrt_alphas_cumprod, t, x0.shape) * x0 +
                extract_into_tensor(sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise)

    # -----------------------
    # JEPA FD Guidance utilities
    # -----------------------
    def _default_jepa_cfg(self):
        return dict(
            enable=False,

            # anomaly (forward-difference, no backward)
            anomaly_fd=False,
            anomaly_n_dir=1,

            # apply window (late steps)
            t_start_ratio=0.7,
            t_end_ratio=1.0,
            every_k=4,

            # update strength
            lambda_=0.05,
            lambda_schedule="constant",  # constant | linear

            # FD params
            fd_eps=1e-3,
            maximize=True,

            # Energy type (Hutchinson trace(J^T J))
            energy_type="hutchinson",  # "hutchinson" (default)

            # Hutchinson params
            hutch_n_samples=4,
            hutch_noise="rademacher",   # "rademacher" | "gaussian"
            hutch_pool="mean",          # encoder output pooling
            hutch_normalize_r=False,    # keep False for unbiased trace estimate
            hutch_seed=1234,            # fix probes across E0/E1 within a step

            # video frame handling
            frame_stride=2,
            max_frames=8,

            # direction mode
            v_mode="score",  # score | random
            normalize_v=True,

            # numeric
            use_fp32=True,

            # optional preprocessing: callable(x_pix)->x_in
            preprocess=None,
        )

    def _jepa_lambda_t(self, cfg, frac: float):
        lam = float(cfg.get("lambda_", 0.0))
        sched = cfg.get("lambda_schedule", "constant")
        if sched == "constant":
            return lam
        elif sched == "linear":
            return lam * frac
        return lam

    def _select_frames_for_jepa(self, x_pix: torch.Tensor, cfg):
        # x_pix: (B,3,T,H,W)
        if x_pix.dim() != 5:
            return x_pix
        B, C, T, H, W = x_pix.shape
        stride = int(cfg.get("frame_stride", 1))
        max_frames = int(cfg.get("max_frames", T))

        idx = torch.arange(0, T, stride, device=x_pix.device)
        if idx.numel() > max_frames:
            idx = idx[:max_frames]
        return x_pix.index_select(dim=2, index=idx)

    @torch.no_grad()
    def _compute_jepa_energy(self, pred_x0: torch.Tensor, cfg):
        """
        pred_x0: latent (B,C,T,H,W)
        return: (B,) per-sample energy

        Hutchinson energy: Tr(J^T J) where J = d(encoder(x))/dx, estimated with Hutchinson probes.
        NOTE: This function temporarily enables autograd internally because Hutchinson JVP needs it.
        """
        # decode latent -> pixel/video
        x_pix = self.model.decode_first_stage(pred_x0)  # (B,3,T,H,W) usually in [-1,1]
        x_pix = self._select_frames_for_jepa(x_pix, cfg)

        preprocess = cfg.get("preprocess", None)
        x_in = preprocess(x_pix) if callable(preprocess) else x_pix

        enc = cfg["encoder_fn"]

        # Hutchinson params
        n_samples = int(cfg.get("hutch_n_samples", 4))
        noise = cfg.get("hutch_noise", "rademacher")
        pool = cfg.get("hutch_pool", "mean")
        normalize_r = bool(cfg.get("hutch_normalize_r", False))
        fd_eps = float(cfg.get("fd_eps", 1e-3))

        # IMPORTANT: hutchinson_trace_jtj uses autograd JVP internally.
        # We are in @torch.no_grad() context overall, so re-enable grad locally.

        with torch.enable_grad():
            E = fd_hutchinson_trace_jtj(
                encoder_fn=enc,
                x=x_in,
                n_samples=n_samples,
                noise=noise,
                pool=pool,
                normalize_r=normalize_r,
                eps_fd=fd_eps,
            )  # (B,)
        return E
