using LinearAlgebra
using OhMyThreads

"""
    colu!(A::StridedMatrix{T}) where {T<:Real}

In-place blocked LU factorization with partial pivoting.
- Overwrites `A` in standard packed form (`L` below diagonal, `U` on/above diagonal).
- Returns a `LinearAlgebra.LU` object.
- Uses `OhMyThreads` for threaded panel/trailing updates.
"""
function colu!(A::StridedMatrix{T}) where {T<:Real}
    m, n = size(A)
    kmax = min(m, n)
    ipiv = Vector{Int}(undef, kmax)
    info = 0
    nb = block_size(T, m, n)

    @inbounds begin
        for k in 1:nb:kmax
            kb = min(nb, kmax - k + 1)
            kend = k + kb - 1

            # Panel factorization on columns k:kend.
            for col in k:kend
                piv = col
                if col < m
                    pivabs = abs(A[col, col])
                    for i in (col + 1):m
                        v = abs(A[i, col])
                        if v > pivabs
                            piv = i
                            pivabs = v
                        end
                    end
                end
                ipiv[col] = piv

                if !iszero(A[piv, col])
                    if piv != col
                        @simd for j in 1:n
                            A[col, j], A[piv, j] = A[piv, j], A[col, j]
                        end
                    end
                    invakk = inv(A[col, col])
                    @simd for i in (col + 1):m
                        A[i, col] *= invakk
                    end
                elseif info == 0
                    info = col
                end

                # Update remaining columns inside the current panel.
                pstart = col + 1
                if pstart <= kend && col < m
                    nt_panel = task_num(kend, col)
                    tforeach(pstart:kend; scheduler = :static, ntasks = nt_panel) do j
                        akj = -A[col, j]
                        @simd ivdep for i in (col + 1):m
                            A[i, j] += A[i, col] * akj
                        end
                    end
                end
            end

            # Trailing update: A[kend+1:m, kend+1:n] -= A[kend+1:m, k:kend] * A[k:kend, kend+1:n]
            rstart = kend + 1
            cstart = kend + 1
            if rstart <= m && cstart <= n
                nt_trail = task_num(n, kend)
                tforeach(cstart:n; scheduler = :static, ntasks = nt_trail) do j
                    for p in k:kend
                        akj = A[p, j]
                        @simd ivdep for i in rstart:m
                            A[i, j] -= A[i, p] * akj
                        end
                    end
                end
            end
        end
    end

    return LU(A, ipiv, info)
end

function task_num(n::Int, k::Int)
    ndiv = 128
    return min(Threads.nthreads(), 1 + fld(n - k, ndiv))
end

function block_size(::Type{T}, m::Int, n::Int) where {T<:Real}
    mn = min(m, n)
    if T === Float16
        return mn >= 512 ? 64 : 32
    end
    return mn >= 512 ? 96 : 48
end
