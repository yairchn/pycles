#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: cdivision=True

import netCDF4 as nc
cimport Grid
cimport ReferenceState
cimport PrognosticVariables
cimport DiagnosticVariables
cimport TimeStepping
from thermodynamic_functions cimport cpm_c, pv_c, pd_c, exner_c
from entropies cimport sv_c, sd_c, s_tendency_c
import numpy as np
import cython
from libc.math cimport fabs, sin, cos, exp
from NetCDFIO cimport NetCDFIO_Stats
cimport ParallelMPI
cimport Lookup
from Thermodynamics cimport LatentHeat, ClausiusClapeyron
import cPickle
from scipy.interpolate import pchip

from fms_forcing_reader import reader

#import pylab as plt
include 'parameters.pxi'

cdef extern from 'advection_interpolation.h':
    double interp_weno3(double phim1, double phi, double phip) nogil

cdef class ForcingGCMMean:
    def __init__(self, namelist, LatentHeat LH, ParallelMPI.ParallelMPI Pa):
        self.file = str(namelist['gcm']['file'])
        self.lat = namelist['gcm']['lat']
        self.lon = namelist['gcm']['lon']
        self.gcm_profiles_initialized = False
        self.t_indx = 0
        return

    @cython.wraparound(True)
    cpdef initialize(self, Grid.Grid Gr,ReferenceState.ReferenceState Ref, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):

        self.coriolis_param = 2.0 * omega * sin(self.lat * pi / 180.0 )

        NS.add_profile('ls_subsidence', Gr, Pa)
        NS.add_profile('ls_dtdt_hadv', Gr, Pa)
        NS.add_profile('ls_dtdt_fino', Gr, Pa)
        NS.add_profile('ls_dtdt_resid', Gr, Pa)
        NS.add_profile('ls_dtdt_fluc', Gr, Pa)
        NS.add_profile('ls_dsdt_hadv', Gr, Pa)
        NS.add_profile('ls_dqtdt_hadv', Gr, Pa)
        NS.add_profile('ls_dqtdt_resid', Gr, Pa)
        NS.add_profile('ls_dqtdt_fluc', Gr, Pa)
        NS.add_profile('ls_subs_dtdt', Gr, Pa)
        NS.add_profile('ls_subs_dsdt', Gr, Pa)
        NS.add_profile('ls_fino_dsdt', Gr, Pa)
        NS.add_profile('ls_subs_dqtdt', Gr, Pa)


        return

    #@cython.wraparound(True)
    cpdef update(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV, TimeStepping.TimeStepping TS,
                 ParallelMPI.ParallelMPI Pa):

        cdef:
            Py_ssize_t gw = Gr.dims.gw
            Py_ssize_t imax = Gr.dims.nlg[0] - gw
            Py_ssize_t jmax = Gr.dims.nlg[1] - gw
            Py_ssize_t kmax = Gr.dims.nlg[2] - gw
            Py_ssize_t istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
            Py_ssize_t jstride = Gr.dims.nlg[2]
            Py_ssize_t i,j,k,ishift,jshift,ijk
            Py_ssize_t u_shift = PV.get_varshift(Gr, 'u')
            Py_ssize_t v_shift = PV.get_varshift(Gr, 'v')
            Py_ssize_t s_shift
            Py_ssize_t thli_shift
            Py_ssize_t qt_shift = PV.get_varshift(Gr, 'qt')
            Py_ssize_t t_shift = DV.get_varshift(Gr, 'temperature')
            Py_ssize_t ql_shift = DV.get_varshift(Gr,'ql')
            Py_ssize_t qi_shift = DV.get_varshift(Gr, 'qi') 
            double pd, pv, qt, qv, p0, rho0, t
            double zmax, weight, weight_half
            double u0_new, v0_new

        if not self.gcm_profiles_initialized:
            self.t_indx = int(TS.t // (3600.0 * 6.0))
            self.gcm_profiles_initialized = True
            Pa.root_print('Updating Time Varying Forcing')

            rdr = reader(self.file, self.lat, self.lon)


            self.ug = rdr.get_interp_profile_old('u_geos', Gr.zp_half)
            self.vg = rdr.get_interp_profile_old('v_geos', Gr.zp_half)
            self.omega_vv = rdr.get_interp_profile_old('omega', Gr.zp_half)
            alpha = rdr.get_interp_profile_old('alpha', Gr.zp_half)
            self.subsidence =  -np.array(self.omega_vv) * alpha / g


            self.temp_dt_hadv  = rdr.get_interp_profile_old('dt_tg_hadv', Gr.zp_half)
            self.temp_dt_fino = rdr.get_interp_profile_old('dt_tg_fino', Gr.zp_half)
            self.temp_dt_resid = rdr.get_interp_profile_old('dt_tg_real1', Gr.zp_half) - rdr.get_interp_profile_old('dt_tg_total', Gr.zp_half)
            self.shum_dt_hadv  = rdr.get_interp_profile_old('dt_qg_hadv', Gr.zp_half)
            self.shum_dt_resid  = rdr.get_interp_profile_old('dt_qg_real1', Gr.zp_half) - rdr.get_interp_profile_old('dt_qg_total', Gr.zp_half)

            temp_dt_vadv_interp = rdr.get_interp_profile('dt_tg_vadv', Gr.zp_half)
            temp_at_zp  = rdr.get_interp_profile('temp', Gr.zp) 
            temp_vadv_pp = np.zeros(np.shape(self.temp_dt_hadv))
            temp_vadv_ppp = np.zeros(np.shape(self.temp_dt_hadv))


            shum_dt_vadv_interp = rdr.get_interp_profile('dt_qg_vadv', Gr.zp_half)
            shum_at_zp = rdr.get_interp_profile('sphum', Gr.zp)
            shum_vadv_pp = np.zeros(np.shape(self.temp_dt_hadv))


            for k in xrange(temp_at_zp.shape[0]-1):
                temp_vadv_pp[k] = temp_dt_vadv_interp[k] + self.temp_dt_fino[k] + ( (temp_at_zp[k+1] - temp_at_zp[k]) * Gr.dims.dxi[2] * Gr.dims.imetl_half[k] + g/cpd)* self.subsidence[k]
                shum_vadv_pp[k] = shum_dt_vadv_interp[k] + ( (shum_at_zp[k+1] - shum_at_zp[k]) * Gr.dims.dxi[2] * Gr.dims.imetl_half[k])* self.subsidence[k]


            for k in xrange(2, temp_at_zp.shape[0]-1):
                tp1 = interp_weno3(temp_at_zp[k-1], temp_at_zp[k], temp_at_zp[k+1])
                tm1 = interp_weno3(temp_at_zp[k-2], temp_at_zp[k-1], temp_at_zp[k])
                temp_vadv_ppp[k] = temp_dt_vadv_interp[k] + self.temp_dt_fino[k] + ( (tp1 - tm1) * Gr.dims.dxi[2] * Gr.dims.imetl_half[k] + g/cpd)* self.subsidence[k]


            #Set some boundary conditions for smoothing
            temp_vadv_pp[:Gr.dims.gw] = temp_vadv_pp[Gr.dims.gw]
            shum_vadv_pp[:Gr.dims.gw] = shum_vadv_pp[Gr.dims.gw]

            # import pylab as plt
            from scipy.signal import savgol_filter
            #Smoothing flucation source terms is helpful becuase taking the vertical derivative of interpolated GCM
            #fields is noisy
            self.temp_dt_fluc = temp_vadv_pp 
            self.shum_dt_fluc = shum_vadv_pp  






            self.rho_gcm =  1.0 / rdr.get_interp_profile('alpha', Gr.zp)
            self.rho_half_gcm = 1.0 / rdr.get_interp_profile('alpha', Gr.zp_half)
            Pa.root_print('Finished updating time varying forcing')

        #Apply Coriolis Forcing
        coriolis_force(&Gr.dims,&PV.values[u_shift],&PV.values[v_shift],&PV.tendencies[u_shift],
                       &PV.tendencies[v_shift],&self.ug[0], &self.vg[0],self.coriolis_param, Ref.u0, Ref.v0  )

        cdef double [:] dtdt_pdv = np.zeros(Gr.dims.npg, dtype=np.double)
        # Apply Subsidence
        if 's' in PV.name_index:
            s_shift = PV.get_varshift(Gr, 's')
            apply_subsidence_temperature(&Gr.dims, &self.rho_gcm[0], &self.rho_half_gcm[0], &self.subsidence[0], &PV.values[qt_shift], &DV.values[t_shift], &PV.tendencies[s_shift])
            compute_pdv_work(&Gr.dims, &self.omega_vv[0], &Ref.p0_half[0], &DV.values[t_shift], &dtdt_pdv[0])
            #apply_subsidence_den(&Gr.dims, &Ref.rho0[0], &Ref.rho0_half[0], &self.subsidence[0], &PV.values[s_shift], &PV.tendencies[s_shift])
        else:
            thli_shift = PV.get_varshift(Gr, 'thli')
            apply_subsidence_temperature_thli(&Gr.dims, &self.rho_gcm[0], &Ref.p0_half[0], &self.rho_half_gcm[0], &self.subsidence[0], &PV.values[qt_shift], &DV.values[t_shift], &PV.tendencies[thli_shift])


        cdef double [:] qt_tend_tmp = np.zeros(Gr.dims.npg, dtype=np.double)
        apply_subsidence(&Gr.dims, &Ref.rho0[0], &Ref.rho0_half[0], &self.subsidence[0], &PV.values[qt_shift], &qt_tend_tmp[0])

        if 's' in PV.name_index:
            s_shift = PV.get_varshift(Gr, 's')
            with nogil:
                for i in xrange(gw,imax):
                    ishift = i * istride
                    for j in xrange(gw,jmax):
                        jshift = j * jstride
                        for k in xrange(gw,kmax):
                            ijk = ishift + jshift + k
                            p0 = Ref.p0_half[k]
                            rho0 = Ref.rho0_half[k]
                            qt = PV.values[qt_shift + ijk]
                            qv = qt - DV.values[ql_shift + ijk] - DV.values[qi_shift + ijk] 
                            pd = pd_c(p0,qt,qv)
                            pv = pv_c(p0,qt,qv)
                            t  = DV.values[t_shift + ijk]

                            PV.tendencies[s_shift + ijk] += (cpm_c(qt) * (self.temp_dt_resid[k] + self.temp_dt_hadv[k] + self.temp_dt_fino[k]*0.0 + self.temp_dt_fluc[k] + dtdt_pdv[ijk]))/t
                            PV.tendencies[s_shift + ijk] += (sv_c(pv,t) - sd_c(pd,t)) * ( self.shum_dt_resid[k] + self.shum_dt_hadv[k]  + qt_tend_tmp[ijk]  + self.shum_dt_fluc[k])
                            PV.tendencies[qt_shift + ijk] += (self.shum_dt_resid[k] + self.shum_dt_hadv[k] + qt_tend_tmp[ijk] + self.shum_dt_fluc[k])
        else:
            thli_shift = PV.get_varshift(Gr, 'thli')
            with nogil:
                for i in xrange(gw,imax):
                    ishift = i * istride
                    for j in xrange(gw,jmax):
                        jshift = j * jstride
                        for k in xrange(gw,kmax):
                            ijk = ishift + jshift + k
                            p0 = Ref.p0_half[k]
                            rho0 = Ref.rho0_half[k]
                            qt = PV.values[qt_shift + ijk]
                            qv = qt - DV.values[ql_shift + ijk]
                            pd = pd_c(p0,qt,qv)
                            pv = pv_c(p0,qt,qv)
                            t  = DV.values[t_shift + ijk]

                            PV.tendencies[thli_shift + ijk] += (self.temp_dt_resid[k] + self.temp_dt_hadv[k] + self.temp_dt_fino[k])/exner_c(Ref.p0_half[k])
                            PV.tendencies[qt_shift + ijk] += (self.shum_dt_resid[k] + self.shum_dt_hadv[k] + qt_tend_tmp[ijk])
                            
        return

    cpdef stats_io(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref,
                 PrognosticVariables.PrognosticVariables PV, DiagnosticVariables.DiagnosticVariables DV,
                   NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):

        cdef:
            Py_ssize_t gw = Gr.dims.gw
            Py_ssize_t kmin = Gr.dims.gw
            Py_ssize_t imax = Gr.dims.nlg[0] - gw
            Py_ssize_t jmax = Gr.dims.nlg[1] - gw
            Py_ssize_t kmax = Gr.dims.nlg[2] - gw
            Py_ssize_t istride = Gr.dims.nlg[1] * Gr.dims.nlg[2]
            Py_ssize_t jstride = Gr.dims.nlg[2]
            Py_ssize_t i,j,k,ishift,jshift,ijk

            #Py_ssize_t s_shift = PV.get_varshift(Gr, 's')
            Py_ssize_t qt_shift = PV.get_varshift(Gr, 'qt')
            Py_ssize_t t_shift = DV.get_varshift(Gr, 'temperature')
            Py_ssize_t ql_shift = DV.get_varshift(Gr,'ql')
            double pd, pv, qt, qv, p0, rho0, t
            double zmax, weight, weight_half

            double [:] qtmean = Pa.HorizontalMean(Gr, &PV.values[qt_shift])
            double [:] tmean = Pa.HorizontalMean(Gr, &DV.values[t_shift])

            double [:] tmp_tendency  = np.zeros((Gr.dims.npg),dtype=np.double,order='c')
            double [:] mean_tendency = np.empty((Gr.dims.nlg[2],),dtype=np.double,order='c')
            double [:] ls_dstd_hadv = np.empty((Gr.dims.nlg[2],),dtype=np.double,order='c')



        apply_subsidence_temperature(&Gr.dims,&self.rho_gcm[0],&self.rho_half_gcm[0],&self.subsidence[0],&PV.values[qt_shift], &DV.values[t_shift],
                         &tmp_tendency[0])
        mean_tendency = Pa.HorizontalMean(Gr,&tmp_tendency[0])
        NS.write_profile('ls_subs_dsdt', mean_tendency[Gr.dims.gw:-Gr.dims.gw], Pa)

        tmp_tendency[:] = 0.0
        apply_subsidence(&Gr.dims,&self.rho_gcm[0],&self.rho_half_gcm[0],&self.subsidence[0], &DV.values[t_shift],
                         &tmp_tendency[0])
        mean_tendency = Pa.HorizontalMean(Gr,&tmp_tendency[0])
        NS.write_profile('ls_subs_dtdt', mean_tendency[Gr.dims.gw:-Gr.dims.gw], Pa)

        tmp_tendency[:] = 0.0
        apply_subsidence(&Gr.dims,&self.rho_gcm[0],&self.rho_half_gcm[0],&self.subsidence[0], &PV.values[qt_shift],
                         &tmp_tendency[0])
        mean_tendency = Pa.HorizontalMean(Gr,&tmp_tendency[0])
        NS.write_profile('ls_subs_dqtdt', mean_tendency[Gr.dims.gw:-Gr.dims.gw], Pa)


        with nogil:
            for k in xrange(kmin, kmax):
                mean_tendency[k]  = mean_tendency[k] * tmean[k] / cpm_c(qtmean[k]) / Ref.rho0_half[k]


        with nogil:

            for i in xrange(Gr.dims.npg):
                tmp_tendency[i] = 0.0

            for i in xrange(gw,imax):
                ishift = i * istride
                for j in xrange(gw,jmax):
                    jshift = j * jstride
                    for k in xrange(gw,kmax):
                        ijk = ishift + jshift + k
                        p0 = Ref.p0_half[k]
                        rho0 = Ref.rho0_half[k]
                        qt = PV.values[qt_shift + ijk]
                        qv = qt - DV.values[ql_shift + ijk]
                        pd = pd_c(p0,qt,qv)
                        pv = pv_c(p0,qt,qv)
                        t  = DV.values[t_shift + ijk]
                        tmp_tendency[ijk] += (cpm_c(qt) * (self.temp_dt_hadv[k]) )/t
                        tmp_tendency[ijk] += (sv_c(pv,t) - sd_c(pd,t)) * (  self.shum_dt_hadv[k] )

        mean_tendency = Pa.HorizontalMean(Gr,&tmp_tendency[0])

        NS.write_profile('ls_subsidence', self.subsidence[Gr.dims.gw:-Gr.dims.gw],Pa)
        NS.write_profile('ls_dtdt_fino',self.temp_dt_fino[Gr.dims.gw:-Gr.dims.gw],Pa)
        NS.write_profile('ls_dtdt_fluc',self.temp_dt_fluc[Gr.dims.gw:-Gr.dims.gw],Pa)
        NS.write_profile('ls_dsdt_hadv', mean_tendency[Gr.dims.gw:-Gr.dims.gw],Pa)
        NS.write_profile('ls_dtdt_hadv', self.temp_dt_hadv[Gr.dims.gw:-Gr.dims.gw],Pa)
        NS.write_profile('ls_dqtdt_hadv', self.shum_dt_hadv[Gr.dims.gw:-Gr.dims.gw],Pa)
        NS.write_profile('ls_dqtdt_fluc', self.shum_dt_fluc[Gr.dims.gw:-Gr.dims.gw], Pa)


        NS.write_profile('ls_dqtdt_resid', self.shum_dt_resid[Gr.dims.gw:-Gr.dims.gw], Pa)
        NS.write_profile('ls_dtdt_resid', self.temp_dt_resid[Gr.dims.gw:-Gr.dims.gw], Pa)


        return

from scipy.interpolate import pchip, interp1d
def interp_pchip(z_out, z_in, v_in, pchip_type=True):
    if pchip_type:
        p = pchip(z_in, v_in, extrapolate=True)
        #p = interp1d(z_in, v_in, kind='linear', fill_value='extrapolate')
        return p(z_out)
    else:
        return np.interp(z_out, z_in, v_in)

cdef coriolis_force(Grid.DimStruct *dims, double *u, double *v, double *ut, double *vt, double *ug, double *vg, double coriolis_param, double u0, double v0 ):

    cdef:
        Py_ssize_t imin = dims.gw
        Py_ssize_t jmin = dims.gw
        Py_ssize_t kmin = dims.gw
        Py_ssize_t imax = dims.nlg[0] -dims.gw
        Py_ssize_t jmax = dims.nlg[1] -dims.gw
        Py_ssize_t kmax = dims.nlg[2] -dims.gw
        Py_ssize_t istride = dims.nlg[1] * dims.nlg[2]
        Py_ssize_t jstride = dims.nlg[2]
        Py_ssize_t ishift, jshift, ijk, i,j,k
        double u_at_v, v_at_u

    with nogil:
        for i in xrange(imin,imax):
            ishift = i*istride
            for j in xrange(jmin,jmax):
                jshift = j*jstride
                for k in xrange(kmin,kmax):
                    ijk = ishift + jshift + k
                    u_at_v = 0.25*(u[ijk] + u[ijk-istride] + u[ijk-istride+jstride] + u[ijk +jstride]) + u0
                    v_at_u = 0.25*(v[ijk] + v[ijk+istride] + v[ijk+istride-jstride] + v[ijk-jstride]) + v0
                    ut[ijk] = ut[ijk] - coriolis_param * (vg[k] - v_at_u)
                    vt[ijk] = vt[ijk] + coriolis_param * (ug[k] - u_at_v)
    return



cdef apply_subsidence_temperature(Grid.DimStruct *dims, double *rho0, double *rho0_half, double *subsidence, double *qt, double* values,  double *tendencies):

    cdef:
        Py_ssize_t imin = dims.gw
        Py_ssize_t jmin = dims.gw
        Py_ssize_t kmin = dims.gw
        Py_ssize_t imax = dims.nlg[0] -dims.gw
        Py_ssize_t jmax = dims.nlg[1] -dims.gw
        Py_ssize_t kmax = dims.nlg[2] -dims.gw -1
        Py_ssize_t istride = dims.nlg[1] * dims.nlg[2]
        Py_ssize_t jstride = dims.nlg[2]
        Py_ssize_t ishift, jshift, ijk, i,j,k
        double dxi = dims.dxi[2]
        double tend
    with nogil:
        for i in xrange(imin,imax):
            ishift = i*istride
            for j in xrange(jmin,jmax):
                jshift = j*jstride
                for k in xrange(kmin,kmax):
                    ijk = ishift + jshift + k
                    if(subsidence[k] < 0):
                        tend = cpm_c(qt[ijk])/values[ijk] *(values[ijk+1] - values[ijk]) * dxi * subsidence[k] * dims.imetl[k]
                    else:
                        tend = cpm_c(qt[ijk])/values[ijk] *(values[ijk] - values[ijk-1]) * dxi * subsidence[k] * dims.imetl[k-1]
                    #+ g / values[ijk] * subsidence[k]
                    tendencies[ijk] -= tend
                for k in xrange(kmax, dims.nlg[2]):
                    ijk = ishift + jshift + k
                    tendencies[ijk] -= tend
    return

cdef apply_subsidence(Grid.DimStruct *dims, double *rho0, double *rho0_half, double *subsidence, double* values,  double *tendencies):

    cdef:
        Py_ssize_t imin = dims.gw
        Py_ssize_t jmin = dims.gw
        Py_ssize_t kmin = dims.gw
        Py_ssize_t imax = dims.nlg[0] -dims.gw
        Py_ssize_t jmax = dims.nlg[1] -dims.gw
        Py_ssize_t kmax = dims.nlg[2] -dims.gw -1
        Py_ssize_t istride = dims.nlg[1] * dims.nlg[2]
        size_t jstride = dims.nlg[2]
        Py_ssize_t ishift, jshift, ijk, i,j,k
        double dxi = dims.dxi[2]
        double tend
    with nogil:
        for i in xrange(imin,imax):
            ishift = i*istride
            for j in xrange(jmin,jmax):
                jshift = j*jstride
                for k in xrange(kmin,kmax):
                    ijk = ishift + jshift + k
                    if(subsidence[k] < 0):
                        tend = (values[ijk+1] - values[ijk]) * dxi * subsidence[k] * dims.imetl[k]
                    else:
                        tend = (values[ijk] - values[ijk-1]) * dxi * subsidence[k] * dims.imetl[k-1]
                    tendencies[ijk] -= tend
                for k in xrange(kmax, dims.nlg[2]):
                    ijk = ishift + jshift + k
                    tendencies[ijk] -= tend

    return

@cython.wraparound(False)
@cython.boundscheck(False)
cdef apply_subsidence_den(Grid.DimStruct *dims, double *rho0, double *rho0_half, double *subsidence, double* values,  double *tendencies):

    cdef:
        Py_ssize_t imin = dims.gw
        Py_ssize_t jmin = dims.gw
        Py_ssize_t kmin = dims.gw
        Py_ssize_t imax = dims.nlg[0] -dims.gw
        Py_ssize_t jmax = dims.nlg[1] -dims.gw
        Py_ssize_t kmax = dims.nlg[2] -dims.gw -1
        Py_ssize_t istride = dims.nlg[1] * dims.nlg[2]
        size_t jstride = dims.nlg[2]
        Py_ssize_t ishift, jshift, ijk, i,j,k
        double dxi = dims.dxi[2]
        double tend
    with nogil:
        for i in xrange(imin,imax):
            ishift = i*istride
            for j in xrange(jmin,jmax):
                jshift = j*jstride
                for k in xrange(kmin,kmax):
                    ijk = ishift + jshift + k
                    if(subsidence[k] < 0):
                        tend = (values[ijk+1]*rho0_half[k+1] - values[ijk]*rho0_half[k]) * dxi * subsidence[k] * dims.imetl[k] / rho0_half[k]
                    else:
                        tend = (values[ijk]*rho0_half[k] - values[ijk-1]*rho0_half[k-1]) * dxi * subsidence[k] * dims.imetl[k-1]/rho0_half[k]
                    tendencies[ijk] -= tend
                for k in xrange(kmax, dims.nlg[2]):
                    ijk = ishift + jshift + k
                    tendencies[ijk] -= tend

    return

cdef compute_pdv_work(Grid.DimStruct *dims, double *omega_vv, double *p0_half, double *T, double* tendency):
    cdef:
        Py_ssize_t imin = dims.gw
        Py_ssize_t jmin = dims.gw
        Py_ssize_t kmin = dims.gw
        Py_ssize_t imax = dims.nlg[0] -dims.gw
        Py_ssize_t jmax = dims.nlg[1] -dims.gw
        Py_ssize_t kmax = dims.nlg[2] -dims.gw
        Py_ssize_t istride = dims.nlg[1] * dims.nlg[2]
        size_t jstride = dims.nlg[2]
        Py_ssize_t ishift, jshift, ijk, i,j,k

    with nogil:
        for i in xrange(imin,imax):
            ishift = i*istride
            for j in xrange(jmin,jmax):
                jshift = j*jstride
                for k in xrange(kmin,kmax):
                    ijk = ishift + jshift + k
                    tendency[ijk] = omega_vv[k] * (Rd * T[ijk])/(p0_half[k] * cpd)
    return


cdef apply_subsidence_temperature_thli(Grid.DimStruct *dims, double *rho0, double *p0_half, double *rho0_half, double *subsidence, double *qt, double* values,  double *tendencies):

    cdef:
        Py_ssize_t imin = dims.gw
        Py_ssize_t jmin = dims.gw
        Py_ssize_t kmin = dims.gw
        Py_ssize_t imax = dims.nlg[0] -dims.gw
        Py_ssize_t jmax = dims.nlg[1] -dims.gw
        Py_ssize_t kmax = dims.nlg[2] -dims.gw -1
        Py_ssize_t istride = dims.nlg[1] * dims.nlg[2]
        Py_ssize_t jstride = dims.nlg[2]
        Py_ssize_t ishift, jshift, ijk, i,j,k
        double dxi = dims.dxi[2]
        double tend
    with nogil:
        for i in xrange(imin,imax):
            ishift = i*istride
            for j in xrange(jmin,jmax):
                jshift = j*jstride
                for k in xrange(kmin,kmax):
                    ijk = ishift + jshift + k
                    if(subsidence[k] < 0):
                        tend = (values[ijk+1] - values[ijk]) * dxi * subsidence[k] * dims.imetl[k]
                    else:
                        tend = (values[ijk] - values[ijk-1]) * dxi * subsidence[k] * dims.imetl[k-1]
                    #+ g / values[ijk] * subsidence[k]
                    tendencies[ijk] -= tend / exner_c(p0_half[k])
                for k in xrange(kmax, dims.nlg[2]):
                    ijk = ishift + jshift + k
                    tendencies[ijk] -= tend / exner_c(p0_half[k])

    return
