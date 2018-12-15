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

#import pylab as plt
include 'parameters.pxi'

cdef class ForcingGCMVarying:
    def __init__(self, namelist, LatentHeat LH, ParallelMPI.ParallelMPI Pa):
        self.file = str(namelist['gcm']['file'])
        self.gcm_profiles_initialized = False
        self.t_indx = 0
        return

    @cython.wraparound(True)
    cpdef initialize(self, Grid.Grid Gr,ReferenceState.ReferenceState Ref, NetCDFIO_Stats NS, ParallelMPI.ParallelMPI Pa):

        fh = open(self.file, 'r')
        input_data_tv = cPickle.load(fh)
        fh.close()

        self.lat = input_data_tv['lat']
        self.coriolis_param = 2.0 * omega * sin(self.lat * pi / 180.0 )

        NS.add_profile('ls_subsidence', Gr, Pa)
        NS.add_profile('ls_dtdt_hadv', Gr, Pa)
        NS.add_profile('ls_dtdt_fino', Gr, Pa)
        NS.add_profile('ls_dsdt_hadv', Gr, Pa)
        NS.add_profile('ls_dqtdt_hadv', Gr, Pa)
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
            Py_ssize_t s_shift = PV.get_varshift(Gr, 's')
            Py_ssize_t qt_shift = PV.get_varshift(Gr, 'qt')
            Py_ssize_t t_shift = DV.get_varshift(Gr, 'temperature')
            Py_ssize_t ql_shift = DV.get_varshift(Gr,'ql')
            double pd, pv, qt, qv, p0, rho0, t
            double zmax, weight, weight_half
            double u0_new, v0_new

        if not self.gcm_profiles_initialized or int(TS.t // (3600.0 * 6.0)) > self.t_indx:
            self.t_indx = int(TS.t // (3600.0 * 6.0))
            self.gcm_profiles_initialized = True
            Pa.root_print('Updating Time Varying Forcing')

            fh = open(self.file, 'r')
            input_data_tv = cPickle.load(fh)
            fh.close()

            zfull = input_data_tv['zfull'][self.t_indx,::-1]
            alpha = input_data_tv['alpha'][self.t_indx,::-1]
            ug = input_data_tv['u_geos'][self.t_indx,::-1]
            vg = input_data_tv['v_geos'][self.t_indx,::-1]
            temp_dt_hadv = input_data_tv['temp_hadv'][self.t_indx,::-1]
            temp_dt_fino = input_data_tv['temp_fino'][self.t_indx,::-1]
            shum_dt_hadv = input_data_tv['dt_qg_hadv'][self.t_indx,::-1]
            temp_dt_resid = input_data_tv['temp_real1'][self.t_indx,::-1] - input_data_tv['temp_total'][self.t_indx,::-1]
            shum_dt_resid = input_data_tv['dt_qg_real1'][self.t_indx,::-1] - input_data_tv['dt_qg_total'][self.t_indx,::-1]
            v_dt_tot = input_data_tv['dt_vg_real1'][self.t_indx,::-1]
            u_dt_tot = input_data_tv['dt_ug_real1'][self.t_indx,::-1]
            omega = input_data_tv['omega'][self.t_indx,::-1]

            temp = input_data_tv['temp'][self.t_indx,::-1]



            self.ug = interp_pchip(Gr.zp_half, zfull, ug)
            self.vg = interp_pchip(Gr.zp_half, zfull, vg)
            self.subsidence = interp_pchip(Gr.zp_half, zfull, -omega * alpha / g)

            self.temp_dt_hadv = interp_pchip(Gr.zp_half, zfull, temp_dt_hadv)
            self.temp_dt_fino = interp_pchip(Gr.zp_half, zfull, temp_dt_fino)
            self.temp_dt_resid = interp_pchip(Gr.zp_half, zfull, temp_dt_resid)
            self.shum_dt_hadv = interp_pchip(Gr.zp_half, zfull, shum_dt_hadv)
            self.shum_dt_resid = interp_pchip(Gr.zp_half, zfull, shum_dt_resid)

            self.rho_gcm = interp_pchip(Gr.zp, zfull, 1.0/alpha)
            self.rho_half_gcm = interp_pchip(Gr.zp_half, zfull, 1.0/alpha)
            self.v_dt_tot = interp_pchip(Gr.zp_half, zfull, v_dt_tot) * 0.0
            self.u_dt_tot = interp_pchip(Gr.zp_half, zfull, u_dt_tot) * 0.0

            Pa.root_print('Finished updating time varying forcing')

            temp = interp_pchip(Gr.zp, zfull, temp)


            #Now preform Galelian transformation
            umean = Pa.HorizontalMean(Gr, &PV.values[u_shift])
            vmean = Pa.HorizontalMean(Gr, &PV.values[v_shift])

            u0_new = 0.0 #(np.max(umean) - np.min(umean))/2.0
            v0_new = 0.0 #(np.max(vmean) - np.min(vmean))/2.0

            with nogil:
                for i in xrange(0,Gr.dims.nlg[0]):
                    ishift = i * istride
                    for j in xrange(0,Gr.dims.nlg[1]):
                        jshift = j * jstride
                        for k in xrange(0,Gr.dims.nlg[2]):
                            ijk = ishift + jshift + k
                            PV.values[u_shift + ijk] -= (u0_new - Ref.u0)
                            PV.values[v_shift + ijk] -= (v0_new - Ref.v0)

            Ref.u0 = u0_new
            Ref.v0 = v0_new

            print "\t Ref.u0 = ", Ref.u0
            print "\t Ref.v0 = ", Ref.v0



        #Apply Coriolis Forcing
        coriolis_force(&Gr.dims,&PV.values[u_shift],&PV.values[v_shift],&PV.tendencies[u_shift],
                      &PV.tendencies[v_shift],&self.ug[0], &self.vg[0],self.coriolis_param, Ref.u0, Ref.v0  )

        # Apply Subsidence
        apply_subsidence_temperature(&Gr.dims, &self.rho_gcm[0], &self.rho_half_gcm[0], &self.subsidence[0], &PV.values[qt_shift], &DV.values[t_shift], &PV.tendencies[s_shift])


        cdef double [:] qt_tend_tmp = np.zeros(Gr.dims.npg, dtype=np.double)
        apply_subsidence(&Gr.dims, &self.rho_gcm[0], &self.rho_half_gcm[0], &self.subsidence[0], &PV.values[qt_shift], &qt_tend_tmp[0])

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

                        PV.tendencies[s_shift + ijk] += (cpm_c(qt) * (self.temp_dt_resid[k] + self.temp_dt_hadv[k] + self.temp_dt_fino[k]))/t
                        PV.tendencies[s_shift + ijk] += (sv_c(pv,t) - sd_c(pd,t)) * ( self.shum_dt_resid[k] + self.shum_dt_hadv[k]  + qt_tend_tmp[ijk]  )
                        PV.tendencies[qt_shift + ijk] +=(self.shum_dt_resid[k] + self.shum_dt_hadv[k] + qt_tend_tmp[ijk])
                        PV.tendencies[u_shift + ijk] += self.u_dt_tot[k]
                        PV.tendencies[v_shift + ijk] += self.v_dt_tot[k]

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

            Py_ssize_t s_shift = PV.get_varshift(Gr, 's')
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
        NS.write_profile('ls_dsdt_hadv', mean_tendency[Gr.dims.gw:-Gr.dims.gw],Pa)
        NS.write_profile('ls_dtdt_hadv', self.temp_dt_hadv[Gr.dims.gw:-Gr.dims.gw],Pa)
        NS.write_profile('ls_dqtdt_hadv', self.shum_dt_hadv[Gr.dims.gw:-Gr.dims.gw],Pa)

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