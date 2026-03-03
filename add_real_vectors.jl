function add_real_vectors(a::AbstractVector{<:Real}, b::AbstractVector{<:Real})
    length(a) == length(b) || throw(DimensionMismatch("vectors must have the same length"))
    return a .+ b
end
