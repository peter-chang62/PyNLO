"""
This is similar to edfa.py, except that two gain fibers can be provided, with a
splicing loss at the splice point. This would likely be used very infrequently
in a fiber amplifier, but can be quite important in a mode-locked fiber laser
where sufficient gain and dispersion balancing must be simultaneously
accounted for.
"""


# %% ----- imports
from edf.re_nlse_joint_5level_wsplice import EDF
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
import copy


def propagate_amp(
    pulse,
    beta_1,
    beta_2,
    edf,
    length,
    Pp,
    n_records=None,
    sum_a_prev=None,
    sum_e_prev=None,
    Pp_prev=None,
    t_shock=None,
    raman_on=False,
):
    edf: EDF
    model = edf.generate_model(
        pulse,
        beta_1,
        beta_2,
        Pp_fwd=Pp,
        sum_a_prev=sum_a_prev,
        sum_e_prev=sum_e_prev,
        Pp_prev=Pp_prev,
        t_shock=t_shock,
        raman_on=raman_on,
    )
    sim = model.simulate(
        length,
        n_records=n_records,
    )
    return model, sim


def amplify(
    p_fwd,
    p_bck,
    beta_1,
    beta_2,
    edf,
    length,
    Pp_fwd,
    Pp_bck,
    t_shock=None,
    raman_on=False,
    n_records=None,
):
    edf: EDF
    edf_1 = EDF(
        alpha_passive=edf.alpha_passive,
        f_r=edf.f_r,
        overlap_p=edf.overlap_p,
        overlap_s=edf.overlap_s,
        n_ion_1=edf._n_ion_1,
        n_ion_2=edf._n_ion_2,
        z_spl=edf.z_spl,
        loss_spl=edf.loss_spl,
        a_eff_1=edf._a_eff_1,
        a_eff_2=edf._a_eff_2,
        gamma_1=edf._gamma_1,
        gamma_2=edf._gamma_2,
        sigma_p=edf.sigma_p,
        sigma_a=edf.sigma_a,
        sigma_e=edf.sigma_e,
    )

    # swap 2 <-> 1, and z_spl -> length - z_spl
    edf_2 = EDF(
        alpha_passive=edf.alpha_passive,
        f_r=edf.f_r,
        overlap_p=edf.overlap_p,
        overlap_s=edf.overlap_s,
        n_ion_1=edf._n_ion_2,
        n_ion_2=edf._n_ion_1,
        z_spl=length - edf.z_spl,
        loss_spl=edf.loss_spl,
        a_eff_1=edf._a_eff_2,
        a_eff_2=edf._a_eff_1,
        gamma_1=edf._gamma_2,
        gamma_2=edf._gamma_1,
        sigma_p=edf.sigma_p,
        sigma_a=edf.sigma_a,
        sigma_e=edf.sigma_e,
    )

    if p_bck is None:
        if Pp_bck == 0:
            model_fwd, sim_fwd = propagate_amp(
                p_fwd, beta_1, beta_2, edf_1, length, Pp_fwd, n_records=n_records
            )
            return model_fwd, sim_fwd, None, None
        else:
            p_bck = p_fwd.copy()
            p_bck.p_t[:] = 0
            bck_seeded = False
    else:
        bck_seeded = True

    done = False
    loop_count = 0
    sum_a_prev = lambda z: 0
    sum_e_prev = lambda z: 0
    Pp_prev = lambda z: 0
    threshold = 1e-3
    while not done:
        model_fwd = edf_1.generate_model(
            p_fwd,
            beta_1,
            beta_2,
            Pp_fwd=Pp_fwd,
            sum_a_prev=sum_a_prev,
            sum_e_prev=sum_e_prev,
            Pp_prev=Pp_prev,
            t_shock=t_shock,
            raman_on=raman_on,
        )
        sim_fwd = model_fwd.simulate(length, n_records=n_records)
        e_p_fwd = sim_fwd.pulse_out.e_p

        sum_a_prev = InterpolatedUnivariateSpline(
            model_fwd.z_record, model_fwd.sum_a_record[::-1]
        )
        sum_e_prev = InterpolatedUnivariateSpline(
            model_fwd.z_record, model_fwd.sum_e_record[::-1]
        )
        Pp_prev = InterpolatedUnivariateSpline(
            model_fwd.z_record, model_fwd.Pp_record[::-1]
        )

        model_bck = edf_2.generate_model(
            p_bck,
            beta_2,
            beta_1,
            Pp_fwd=Pp_bck,
            sum_a_prev=sum_a_prev,
            sum_e_prev=sum_e_prev,
            Pp_prev=Pp_prev,
            t_shock=t_shock,
            raman_on=raman_on,
        )
        if bck_seeded:
            sim_bck = model_bck.simulate(length, n_records=n_records)
            e_p_bck = sim_bck.pulse_out.e_p

            sum_a_prev = InterpolatedUnivariateSpline(
                model_bck.z_record, model_bck.sum_a_record[::-1]
            )
            sum_e_prev = InterpolatedUnivariateSpline(
                model_bck.z_record, model_bck.sum_e_record[::-1]
            )
            Pp_prev = InterpolatedUnivariateSpline(
                model_bck.z_record, model_bck.Pp_record[::-1]
            )

            # for return results
            Pp_total = sim_fwd.Pp + sim_bck.Pp[::-1]
            sim_fwd.Pp = Pp_total
            sim_bck.Pp = Pp_total
        else:
            rk45 = model_bck.mode.rk45_Pp
            t = [rk45.t]
            y = [rk45.y[0]]
            loss_applied = False
            while rk45.t < length:
                model_bck.mode.z = rk45.t  # update z-dependent parameter
                rk45.step()
                if rk45.t > edf_2.z_spl:
                    if not loss_applied:
                        rk45.y *= edf_2.loss_spl
                        loss_applied = True
                t.append(rk45.t)
                y.append(rk45.y[0])
            t = np.asarray(t)
            y = np.asarray(y)

            sum_a_prev = lambda z: 0
            sum_e_prev = lambda z: 0
            Pp_prev = InterpolatedUnivariateSpline(t, y[::-1])

            # no error to calculate
            e_p_bck = 1e-20  # avoid divide 0 errors

            # for return results
            sim_bck = None
            sim_fwd.Pp += Pp_prev(sim_fwd.z)

        # book keeping
        if loop_count == 0:
            e_p_fwd_old = e_p_fwd
            e_p_bck_old = e_p_bck
            loop_count += 1
            continue

        error_fwd = abs(e_p_fwd - e_p_fwd_old) / e_p_fwd
        error_bck = abs(e_p_bck - e_p_bck_old) / e_p_bck
        e_p_fwd_old = e_p_fwd
        e_p_bck_old = e_p_bck
        if error_fwd < threshold and error_bck < threshold:
            done = True
        print(loop_count, error_fwd, error_bck)
        loop_count += 1

    return model_fwd, sim_fwd, model_bck, sim_bck
