#= src/lapack.jl
=#

module _LAPACK
import LinearAlgebra
import LinearAlgebra.BLAS: @blasfunc
using LinearAlgebra: BlasFloat, BlasInt, chkstride1, require_one_based_indexing
using LinearAlgebra.LAPACK: chklapackerror
import Base: Nothing
const liblapack = LinearAlgebra.BLAS.liblapack

for (geqrf, gelqf, orgqr, orglq, elty) in
      ((:sgeqrf_, :sgelqf_, :sorgqr_, :sorglq_, :Float32   ),
       (:dgeqrf_, :dgelqf_, :dorgqr_, :dorglq_, :Float64   ),
       (:cgeqrf_, :cgelqf_, :cungqr_, :cunglq_, :ComplexF32 ),
       (:zgeqrf_, :zgelqf_, :zungqr_, :zunglq_, :ComplexF64))
  @eval begin
    function geqrf!(
        A::AbstractMatrix{$elty}, tau::Vector{$elty}, work::Vector{$elty})
      chkstride1(A)
      m, n  = size(A)
      k     = min(m, n)
      tau   = length(tau) < k ? Array{$elty}(undef, k) : tau
      lwork = BlasInt(-1)
      info  = Ref{BlasInt}()
      for i = 1:2
        ccall((@blasfunc($geqrf), liblapack), Nothing,
              (Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
               Ptr{$elty}, Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt}),
              m, n, A, max(1,stride(A,2)),
              tau, work, lwork, info)
        if i == 1
          lwork = BlasInt(real(work[1]))
          work  = length(work) < lwork ? Array{$elty}(undef, lwork) : work
          lwork = length(work)
        end
      end
      A, tau, work
    end

    function gelqf!(
        A::AbstractMatrix{$elty}, tau::Vector{$elty}, work::Vector{$elty})
      chkstride1(A)
      m, n  = size(A)
      k     = min(m, n)
      tau   = length(tau) < k ? Array{$elty}(undef, k) : tau
      lwork = BlasInt(-1)
      info  = Ref{BlasInt}()
      for i = 1:2
        ccall((@blasfunc($gelqf), liblapack), Nothing,
              (Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
               Ptr{$elty}, Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt}),
              m, n, A, max(1,stride(A,2)),
              tau, work, lwork, info)
        if i == 1
            lwork = BlasInt(real(work[1]))
            work  = length(work) < lwork ? Array{$elty}(undef, lwork) : work
            lwork = length(work)
        end
      end
      A, tau, work
    end

    function orglq!(
        A::AbstractMatrix{$elty}, tau::Vector{$elty}, k::Integer,
        work::Vector{$elty})
      chkstride1(A)
      n = size(A, 2)
      m = min(n, size(A, 1))
      0 <= k <= min(m, length(tau)) || throw(DimensionMismatch)
      lwork = BlasInt(-1)
      info  = Ref{BlasInt}()
      for i = 1:2
          ccall((@blasfunc($orglq), liblapack), Nothing,
                (Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty},
                 Ref{BlasInt}, Ptr{$elty}, Ptr{$elty}, Ref{BlasInt},
                 Ptr{BlasInt}),
                m, n, k, A,
                max(1,stride(A,2)), tau, work, lwork,
                info)
          if i == 1
              lwork = BlasInt(real(work[1]))
              work  = length(work) < lwork ? Array{$elty}(undef, lwork) : work
              lwork = length(work)
          end
      end
      A, tau, work
    end

    function orgqr!(
        A::AbstractMatrix{$elty}, tau::Vector{$elty}, k::Integer,
        work::Vector{$elty})
      chkstride1(A)
      m = size(A, 1)
      n = min(m, size(A,2))
      0 <= k <= min(n, length(tau)) || throw(DimensionMismatch)
      lwork = BlasInt(-1)
      info  = Ref{BlasInt}()
      for i = 1:2
        ccall((@blasfunc($orgqr), liblapack), Nothing,
              (Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty},
               Ref{BlasInt}, Ptr{$elty}, Ptr{$elty}, Ref{BlasInt},
               Ptr{BlasInt}),
              m, n, k, A,
              max(1,stride(A,2)), tau, work, lwork,
              info)
        if i == 1
          lwork = BlasInt(real(work[1]))
          work  = length(work) < lwork ? Array{$elty}(undef, lwork) : work
          lwork = length(work)
        end
      end
      A, tau, work
    end
  end
end

for (laqps, elty, relty) in ((:slaqps_, :Float32,    :Float32),
                             (:dlaqps_, :Float64,    :Float64),
                             (:claqps_, :ComplexF32,  :Float32),
                             (:zlaqps_, :ComplexF64, :Float64))
  @eval begin
    function laqps!(
        offset::BlasInt, nb::BlasInt, kb::Ref{BlasInt},
        A::AbstractMatrix{$elty},
        jpvt::AbstractVector{BlasInt}, tau::AbstractVector{$elty},
        vn1::AbstractVector{$relty}, vn2::AbstractVector{$relty},
        auxv::AbstractVector{$elty}, F::AbstractVector{$elty})
      m, n = size(A)
      ccall(
        (@blasfunc($laqps), liblapack), Nothing,
        (Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt}, Ptr{BlasInt},
         Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt}, Ptr{$elty},
         Ptr{$relty}, Ptr{$relty}, Ptr{$elty}, Ptr{$elty}, Ref{BlasInt}),
        m, n, offset, nb, kb,
        A, max(1,stride(A,2)), jpvt, tau,
        vn1, vn2, auxv, F, max(1,n))
    end
  end
end

for (geqp3rk, elty, relty) in ((:sgeqp3rk_,:Float32,:Float32),
                               (:dgeqp3rk_,:Float64,:Float64),
                               (:cgeqp3rk_,:ComplexF32,:Float32),
                               (:zgeqp3rk_,:ComplexF64,:Float64))
  @eval begin
    function geqp3rk!(
      A::AbstractMatrix{$elty},
      nrhs::BlasInt,
      kmax::BlasInt,
      abstol::BlasFloat,
      reltol::BlasFloat,
      jpvt::AbstractVector{BlasInt}, 
      tau::AbstractVector{$elty}
    )
      require_one_based_indexing(A, jpvt, tau)
      chkstride1(A,jpvt,tau)
      m,n = size(A)
      if length(tau) != min(m,n)
          throw(DimensionMismatch(lazy"tau has length $(length(tau)), but needs length $(min(m,n))"))
      end
      if length(jpvt) != n
          throw(DimensionMismatch(lazy"jpvt has length $(length(jpvt)), but needs length $n"))
      end
      lda = stride(A,2)
      if lda == 0
          return A, tau, jpvt
      end # Early exit
      work  = Vector{$elty}(undef, 1)
      lwork = BlasInt(-1)
      iwork = Vector{BlasInt}(undef, n - 1)
      cmplx = eltype(A)<:Complex
      if cmplx
          rwork = Vector{$relty}(undef, 2n)
      end
      info = Ref{BlasInt}()
      k = Ref{BlasInt}()
      maxc2nrmk = Ref{$relty}()
      relmaxc2nrmk = Ref{$relty}()
      for i = 1:2  # first call returns lwork as work[1]
          if cmplx
              ccall((@blasfunc($geqp3rk), liblapack), Cvoid,
                      (
                        Ref{BlasInt}, #m
                        Ref{BlasInt}, #n
                        Ref{BlasInt}, #nrhs
                        Ref{BlasInt},#kmax
                        Ref{$relty},  #abstol
                        Ref{$relty},  #reltol
                        Ptr{$elty},   #A
                        Ref{BlasInt}, #lda
                        Ptr{BlasInt}, #k
                        Ptr{$relty}, #maxc2nrmk
                        Ptr{$relty}, #relmaxc2nrmk
                        Ptr{BlasInt}, #jpvt -> jpiv
                        Ptr{$elty},   #tau
                        Ptr{$elty},   #work
                        Ref{BlasInt}, #lwork
                        Ptr{$relty}, #rwork
                        Ptr{BlasInt}, #iwork
                        Ptr{BlasInt}  #info
                      ),
                      m, n, nrhs,
                      kmax, abstol, reltol,
                      A, lda,
                      k, maxc2nrmk, relmaxc2nrmk,
                      jpvt, tau, work,
                      lwork, rwork, iwork, info)
          else
              #println("running ccall")
              ccall((@blasfunc($geqp3rk), liblapack), Cvoid,
                    (
                      Ref{BlasInt}, #m
                      Ref{BlasInt}, #n
                      Ref{BlasInt}, #nrhs 
                      Ref{BlasInt},#kmax  
                      Ref{$elty},  #abstol 
                      Ref{$elty},  #reltol  
                      Ptr{$elty},   #A
                      Ref{BlasInt}, #lda
                      Ref{BlasInt}, #k  
                      Ptr{$elty}, #maxc2nrmk  
                      Ptr{$elty}, #relmaxc2nrmk  
                      Ptr{BlasInt}, #jpvt -> jpiv
                      Ptr{$elty},   #tau
                      Ptr{$elty},   #work
                      Ref{BlasInt}, #lwork
                      Ptr{BlasInt}, #iwork 
                      Ptr{BlasInt}  #info
                    ),
                    m, n, nrhs,
                    kmax, abstol, reltol,
                    A, lda,
                    k, maxc2nrmk, relmaxc2nrmk,
                    jpvt, tau, work,
                    lwork, iwork, info)
          end
          chklapackerror(info[])
          if i == 1
              lwork = BlasInt(real(work[1]))
              resize!(work, lwork)
          end
      end
      return A, k[], tau, jpvt
    end
  end
end


end  # module
