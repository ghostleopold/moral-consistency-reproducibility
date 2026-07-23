module MoralConsistency
using DataFrames
using Roots
using ProgressMeter
using StaticArrays
using CSV
using ProgressMeter
using Base.Threads

export d_int_to_bin

const GAME_TYPES = ["PD", "SG", "SH"] # prisoner's dilemma, snow-drift game, stag-hunt
const STRATEGY_SPACES = ["complete", "toy", "test"] # potentially skip mirrors later

# Type aliases for clarity
const BinaryVector = Vector{Int}
const PayoffTuple = Tuple{Float64, Float64, Float64, Float64}

# Precomputed tables for binary operations
const PRECOMPUTED_INDICES = [1+l+2*k+4*j+8*i for i in 0:1, j in 0:1, k in 0:1, l in 0:1]
const PRECOMPUTED_P_INDICES = [1+j+2*i for i in 0:1, j in 0:1]


# Use Ref() to store a mutable reference to the thread caches
const THREAD_CACHES = Ref{Vector{Dict{Tuple{Int,Int,Float64,Float64}, Float64}}}()

function initialize_thread_caches()
    THREAD_CACHES[] = [Dict{Tuple{Int,Int,Float64,Float64}, Float64}() for _ in 1:Threads.nthreads()]
    println("Initialized THREAD_CACHES for ", Threads.nthreads(), " threads.")
end


struct Strategy
    p::BitVector
    d::BitVector
    p_int::Int
    d_int::Int
    
    function Strategy(p_int::Int, d_int::Int)
        p = BitVector(digits(p_int, base=2, pad=4))
        d = BitVector(digits(d_int, base=2, pad=16))
        new(p, d, p_int, d_int)
    end
end

###-- Binary to int and viceversa

function d_int_to_bin(d_int)
    return digits(d_int,base=2,pad=16)
end

function p_int_to_bin(p_int)
    return digits(p_int,base=2,pad=4)
end

function p_bin_to_int(p)
    return digits_to_int(p,4)
end

function d_bin_to_int(d)
    return digits_to_int(d,16)

end 


# Optimized helper functions
@inline function getd(s::Strategy, i::Int, j::Int, k::Int, l::Int)::Int
    @inbounds return s.d[PRECOMPUTED_INDICES[i+1,j+1,k+1,l+1]]
end

@inline function getp(s::Strategy, i::Int, j::Int)::Int
    @inbounds return s.p[PRECOMPUTED_P_INDICES[i+1,j+1]]
end

# convert array of digits (0,1) to integer
function digits_to_int(di,pad)
	n = 0
	for i in 1:pad
		n = n + di[i]*2^(i-1)
	end
	n
end



###-- END Binary to int and viceversa


function complete_strategy_space()
    ans = []
    for d in 0:65535 # all norms
        for p in 0:15 # all action rules
            push!(ans, (p, d))
        end
    end
    return ans   
end

function  toy_strategy_space()
    ans = []
    for d in vcat([0, 53196, 100, 50124],[1000:1002;]) # some norms
        for p in [8:11;] # some action rules
            push!(ans, (p, d))
        end
    end
    return ans
end

function test_strategy_space()
    ans = [
        (14, 45159), (3, 46049), (9, 12318), (5, 31480), (8, 57939), (3, 31560), (2, 30290), (6, 24245), 
        (8, 35029), (12, 58313), (6, 38223), (14, 18335), (6, 31102), (6, 64502), (1, 7737), (7, 65236), 
        (2, 39357), (13, 62663), (7, 49065), (14, 8697), (5, 61813), (10, 27155), (11, 58472), (3, 24427), 
        (12, 63419), (12, 50517), (6, 65509), (2, 40483), (13, 34207), (13, 36057), (14, 34979), (13, 58805), 
        (4, 20571), (3, 7128), (11, 52319), (9, 1877), (6, 31392), (12, 7427), (6, 4017), (4, 19715), 
        (4, 14625), (1, 6948), (5, 56591), (9, 24731), (12, 43411), (14, 22195), (5, 48638), (2, 13791), 
        (4, 15770), (7, 45219)
    ]
    return ans
end


"""
Computes the reputation equilibrium for the resident strategy
"""


function δ(p::BinaryVector, d::BinaryVector, χ::Float64, ϵ::Float64, i::Int, j::Int)::Float64
    return χ + (1-2*χ)*(
        ((1-ϵ)^2)*getd(d,i,j,getp(p,i,j),getp(p,j,i)) + 
        ϵ*(1-ϵ)*(getd(d,i,j,0,getp(p,j,i))+getd(d,i,j,getp(p,i,j),0)) + 
        (ϵ^2)*getd(d,i,j,0,0)
    )
end

function get_resident_reputation_eq(s::Strategy, χ::Float64, ϵ::Float64)::Float64
    cache = THREAD_CACHES[][Threads.threadid()]  # Get the cache for this thread
    cache_key = (s.p_int, s.d_int, χ, ϵ)
    
    if haskey(cache, cache_key)
        return cache[cache_key]
    end

    # Resident plays against resident
    δ00 = χ + (1-2χ)*((1-ϵ)^2*getd(s,0,0,getp(s,0,0),getp(s,0,0)) + 
          ϵ*(1-ϵ)*(getd(s,0,0,0,getp(s,0,0))+getd(s,0,0,getp(s,0,0),0)) + 
          ϵ^2*getd(s,0,0,0,0))
    
    δ11 = χ + (1-2χ)*((1-ϵ)^2*getd(s,1,1,getp(s,1,1),getp(s,1,1)) + 
          ϵ*(1-ϵ)*(getd(s,1,1,0,getp(s,1,1))+getd(s,1,1,getp(s,1,1),0)) + 
          ϵ^2*getd(s,1,1,0,0))
    
    δ01 = χ + (1-2χ)*((1-ϵ)^2*getd(s,0,1,getp(s,0,1),getp(s,1,0)) + 
          ϵ*(1-ϵ)*(getd(s,0,1,0,getp(s,1,0))+getd(s,0,1,getp(s,0,1),0)) + 
          ϵ^2*getd(s,0,1,0,0))
    
    δ10 = χ + (1-2χ)*((1-ϵ)^2*getd(s,1,0,getp(s,1,0),getp(s,0,1)) + 
          ϵ*(1-ϵ)*(getd(s,1,0,0,getp(s,0,1))+getd(s,1,0,getp(s,1,0),0)) + 
          ϵ^2*getd(s,1,0,0,0))
    
    a = δ00 + δ11 - δ01 - δ10
    b = δ01 + δ10 - 2δ00 - 1
    c = δ00
    
    result = calc_eq_from_coefficients(a, b, c)
    

    # Store in thread-local cache
    cache[cache_key] = result

    return result
end


function resident_reputation_eq_int(p_int,d_int,χ,ϵ)
    p = p_int_to_bin(p_int)
    d = d_int_to_bin(d_int)
    return resident_reputation_eq(p,d,χ,ϵ)
end




## ESS functions

@inline function getpayoff(π::NTuple{4,Float64}, i::Int, j::Int)::Float64
    @inbounds return π[1+j+2*i]
end

function δm(pm::BinaryVector, p::BinaryVector, d::BinaryVector, χ::Float64, ϵ::Float64, i::Int, j::Int)::Float64
    return χ + (1-2*χ) * (
        (1-ϵ)^2*getd(d,i,j,getp(pm,i,j),getp(p,j,i)) + 
        ϵ*(1-ϵ)*(getd(d,i,j,0,getp(p,j,i)) + getd(d,i,j,getp(pm,i,j),0)) + 
        ϵ^2*getd(d,i,j,0,0)
    )
end


# Optimized reputation equilibrium calculation with caching
function get_reputation_eq(s::Strategy, χ::Float64, ϵ::Float64)::Float64
    cache_key = (s.p_int, s.d_int, χ, ϵ)
    
    @inbounds if haskey(REPUTATION_CACHE, cache_key)
        return REPUTATION_CACHE[cache_key]
    end
    
    # Unrolled and optimized δ calculations
    δ00 = χ + (1-2χ)*((1-ϵ)^2*getd(s,0,0,getp(s,0,0),getp(s,0,0)) + 
          ϵ*(1-ϵ)*(getd(s,0,0,0,getp(s,0,0))+getd(s,0,0,getp(s,0,0),0)) + 
          ϵ^2*getd(s,0,0,0,0))
    
    δ11 = χ + (1-2χ)*((1-ϵ)^2*getd(s,1,1,getp(s,1,1),getp(s,1,1)) + 
          ϵ*(1-ϵ)*(getd(s,1,1,0,getp(s,1,1))+getd(s,1,1,getp(s,1,1),0)) + 
          ϵ^2*getd(s,1,1,0,0))
    
    δ01 = χ + (1-2χ)*((1-ϵ)^2*getd(s,0,1,getp(s,0,1),getp(s,1,0)) + 
          ϵ*(1-ϵ)*(getd(s,0,1,0,getp(s,1,0))+getd(s,0,1,getp(s,0,1),0)) + 
          ϵ^2*getd(s,0,1,0,0))
    
    δ10 = χ + (1-2χ)*((1-ϵ)^2*getd(s,1,0,getp(s,1,0),getp(s,0,1)) + 
          ϵ*(1-ϵ)*(getd(s,1,0,0,getp(s,0,1))+getd(s,1,0,getp(s,1,0),0)) + 
          ϵ^2*getd(s,1,0,0,0))
    
    a = δ00 + δ11 - δ01 - δ10
    b = δ01 + δ10 - 2δ00 - 1
    c = δ00
    
    # Optimized quadratic solution
    discriminant = b^2 - 4a*c
    if abs(a) < 1e-10  # Linear case
        result = -c/b
    else
        result = (-b - sqrt(discriminant))/(2a)
    end
    
    # Bound result to [0,1]
    result = clamp(result, 0.0, 1.0)
    
    REPUTATION_CACHE[cache_key] = result
    return result
end

function get_mutant_reputation_eq(mutant::Strategy, resident::Strategy, χ::Float64, ϵ::Float64)::Float64
    # Get resident equilibrium first
    g = get_resident_reputation_eq(resident, χ, ϵ)
    
    # Calculate all δm terms for mutant playing against resident
    δm00 = χ + (1-2χ)*((1-ϵ)^2*getd(resident,0,0,getp(mutant,0,0),getp(resident,0,0)) + 
           ϵ*(1-ϵ)*(getd(resident,0,0,0,getp(resident,0,0))+getd(resident,0,0,getp(mutant,0,0),0)) + 
           ϵ^2*getd(resident,0,0,0,0))
    
    δm01 = χ + (1-2χ)*((1-ϵ)^2*getd(resident,0,1,getp(mutant,0,1),getp(resident,1,0)) + 
           ϵ*(1-ϵ)*(getd(resident,0,1,0,getp(resident,1,0))+getd(resident,0,1,getp(mutant,0,1),0)) + 
           ϵ^2*getd(resident,0,1,0,0))
    
    δm10 = χ + (1-2χ)*((1-ϵ)^2*getd(resident,1,0,getp(mutant,1,0),getp(resident,0,1)) + 
           ϵ*(1-ϵ)*(getd(resident,1,0,0,getp(resident,0,1))+getd(resident,1,0,getp(mutant,1,0),0)) + 
           ϵ^2*getd(resident,1,0,0,0))
    
    δm11 = χ + (1-2χ)*((1-ϵ)^2*getd(resident,1,1,getp(mutant,1,1),getp(resident,1,1)) + 
           ϵ*(1-ϵ)*(getd(resident,1,1,0,getp(resident,1,1))+getd(resident,1,1,getp(mutant,1,1),0)) + 
           ϵ^2*getd(resident,1,1,0,0))
    
    # Calculate numerator and denominator for mutant reputation
    num = δm00 + (δm01 - δm00)*g
    den = 1 + δm00 - δm10 - (δm00 + δm11 - δm01 - δm10)*g
    
    return clamp(num/den, 0.0, 1.0)
end

# Helper for reputation equilibrium calculation
function calc_eq_from_coefficients(a::Float64, b::Float64, c::Float64)::Float64
    if abs(a) < 1e-10  # Linear case
        result = -c/b
    else
        discriminant = b^2 - 4a*c
        result = (-b - sqrt(discriminant))/(2a)
    end
    return clamp(result, 0.0, 1.0)
end



# Separate fitness calculations for resident and mutant
function calc_resident_fitness(s::Strategy, χ::Float64, ϵ::Float64, π::PayoffTuple)::Float64
    g = get_resident_reputation_eq(s, χ, ϵ)
    
    # Precompute common terms
    ϵ_sq = ϵ^2
    ϵ_term = ϵ*(1-ϵ)
    pure_term = (1-ϵ)^2
    
    # Calculate ω terms for resident-resident interactions
    ω11 = pure_term*getpayoff(π,getp(s,1,1),getp(s,1,1)) + 
          ϵ_term*(getpayoff(π,0,getp(s,1,1)) + getpayoff(π,getp(s,1,1),0)) + 
          ϵ_sq*getpayoff(π,0,0)
    
    ω01 = pure_term*getpayoff(π,getp(s,0,1),getp(s,1,0)) + 
          ϵ_term*(getpayoff(π,0,getp(s,1,0)) + getpayoff(π,getp(s,0,1),0)) + 
          ϵ_sq*getpayoff(π,0,0)
    
    ω10 = pure_term*getpayoff(π,getp(s,1,0),getp(s,0,1)) + 
          ϵ_term*(getpayoff(π,0,getp(s,0,1)) + getpayoff(π,getp(s,1,0),0)) + 
          ϵ_sq*getpayoff(π,0,0)
    
    ω00 = pure_term*getpayoff(π,getp(s,0,0),getp(s,0,0)) + 
          ϵ_term*(getpayoff(π,0,getp(s,0,0)) + getpayoff(π,getp(s,0,0),0)) + 
          ϵ_sq*getpayoff(π,0,0)
    
    g_sq = g^2
    g_comp = 1.0 - g
    
    return ω11*g_sq + (ω01 + ω10)*g*g_comp + ω00*g_comp^2
end



function calc_mutant_fitness(mutant::Strategy, resident::Strategy, χ::Float64, ϵ::Float64, π::PayoffTuple)::Float64
    g = get_resident_reputation_eq(resident, χ, ϵ)
    h = get_mutant_reputation_eq(mutant, resident, χ, ϵ)
    
    # Precompute common terms
    ϵ_sq = ϵ^2
    ϵ_term = ϵ*(1-ϵ)
    pure_term = (1-ϵ)^2
    
    # Calculate ω terms for mutant-resident interactions
    ωm11 = pure_term*getpayoff(π,getp(mutant,1,1),getp(resident,1,1)) + 
           ϵ_term*(getpayoff(π,0,getp(resident,1,1)) + getpayoff(π,getp(mutant,1,1),0)) + 
           ϵ_sq*getpayoff(π,0,0)
    
    ωm10 = pure_term*getpayoff(π,getp(mutant,1,0),getp(resident,0,1)) + 
           ϵ_term*(getpayoff(π,0,getp(resident,0,1)) + getpayoff(π,getp(mutant,1,0),0)) + 
           ϵ_sq*getpayoff(π,0,0)
    
    ωm01 = pure_term*getpayoff(π,getp(mutant,0,1),getp(resident,1,0)) + 
           ϵ_term*(getpayoff(π,0,getp(resident,1,0)) + getpayoff(π,getp(mutant,0,1),0)) + 
           ϵ_sq*getpayoff(π,0,0)
    
    ωm00 = pure_term*getpayoff(π,getp(mutant,0,0),getp(resident,0,0)) + 
           ϵ_term*(getpayoff(π,0,getp(resident,0,0)) + getpayoff(π,getp(mutant,0,0),0)) + 
           ϵ_sq*getpayoff(π,0,0)
    
    return h*(g*ωm11 + (1-g)*ωm10) + (1-h)*(g*ωm01 + (1-g)*ωm00)
end

function is_ESS(s::Strategy, χ::Float64, ϵ::Float64, π::PayoffTuple)::Bool
    W_res = calc_resident_fitness(s, χ, ϵ, π)
    
    @inbounds for pm_int in 0:15
        if pm_int != s.p_int
            mutant = Strategy(pm_int, s.d_int)
            W_mut = calc_mutant_fitness(mutant, s, χ, ϵ, π)
            if W_mut >= W_res
                return false
            end
        end
    end
    return true
end


function is_ESS_int(p_int,d_int,χ,ϵ,π)
    s = Strategy(p_int, d_int)
    return is_ESS(s, χ, ϵ, π)    
end

function is_ESS_int_spread(p_int,d_int,χ,ϵ,P,T,S,R)
    p = p_int_to_bin(p_int)
    d = d_int_to_bin(d_int)
	return is_ESS(p,d,χ,ϵ,(P,T,S,R))
end


function stability_index(strategy_space, df_payoffs, χ, ϵ)
    # Pre-group the payoffs by game type
    grouped_payoffs = groupby(df_payoffs, :game)
    game_types = keys(grouped_payoffs)
    
    # Pre-allocate results array
    n_strategies = length(strategy_space)
    n_games = length(game_types)
    total_rows = n_strategies * n_games
    
    # Pre-allocate vectors for final DataFrame
    p_vec = Vector{Int}(undef, total_rows)
    d_vec = Vector{Int}(undef, total_rows)
    game_vec = Vector{String}(undef, total_rows)
    stab_index_vec = Vector{Float64}(undef, total_rows)
    sample_vec = Vector{Int}(undef, total_rows)
    
    # Create a single progress bar with the total number of computations
    println("Processing stability indices...")
    progress = Progress(total_rows; desc="Computing: ", showspeed=true)
    
    # Process each game group once
    idx = 1
    for group_key in game_types
        game = group_key.game  # Extract the actual game name from the GroupKey
        game_df = grouped_payoffs[group_key]
        
        # Convert to static vectors for faster access
        payoff_data = [@SVector [row.P, row.T, row.S, row.R] for row in eachrow(game_df)]
        n_samples = length(payoff_data)
        
        # Process each strategy for this game
        for (p, d) in strategy_space
            # Compute stability index for current strategy and game
            stable_count = 0
            @inbounds for i in 1:n_samples
                payoff = payoff_data[i]
                stable_count += is_ESS_int(p, d, χ, ϵ, 
                    (payoff[1], payoff[2], payoff[3], payoff[4]))
            end
            
            # Store results
            p_vec[idx] = p
            d_vec[idx] = d
            game_vec[idx] = game
            stab_index_vec[idx] = stable_count / n_samples
            sample_vec[idx] = n_samples
            
            # Update progress bar
            ProgressMeter.update!(progress, idx)
            idx += 1
        end
    end
    
    # Finish the progress bar
    ProgressMeter.finish!(progress)
    
    # Create final DataFrame
    return DataFrame(
        p = p_vec,
        d = d_vec,
        game = game_vec,
        stability_index = stab_index_vec,
        sampled_points = sample_vec
    )
end


##------- Global properties --------##


"""
Computes the cooperation index
"""
function cooperation_index(p_int,d_int,χ,ϵ)
	g = get_resident_reputation_eq(Strategy(p_int,d_int),χ,ϵ)
	p = p_int_to_bin(p_int)
	return (1-ϵ)*(g*(g*p[4]+(1-g)*p[3]) + (1-g)*(g*p[2]+(1-g)*p[1]))
end


# is norm d of a given class? 
# all() is used to check if all of the values in the given collection are true
# 0th order: all bits are the same
is_class_0(d) = all(x->x==d[1],d)
# 1st order: only one bit is different
is_class_1l(d) = all(x->x==d[1:2:16][1],d[1:2:16]) && all(x->x==d[2:2:16][1],d[2:2:16]) # only bit l is different
is_class_1k(d) = all(x->x==d[[1,2,5,6,9,10,13,14]][1],d[[1,2,5,6,9,10,13,14]]) && all(x->x==d[[3,4,7,8,11,12,15,16]][1],d[[3,4,7,8,11,12,15,16]]) # only bit k is different
is_class_1j(d) = all(x->x==d[[1,2,3,4,9,10,11,12]][1],d[[1,2,3,4,9,10,11,12]]) && all(x->x==d[[5,6,7,8,13,14,15,16]][1],d[[5,6,7,8,13,14,15,16]]) # only bit j is different
is_class_1i(d) = all(x->x==d[1:8][1],d[1:8]) && all(x->x==d[9:16][1],d[9:16]) # only bit i is different
is_class_1(d) = is_class_1l(d) || is_class_1k(d) || is_class_1j(d) || is_class_1i(d)
# 2nd order
is_class_2kl(d) = all(x->x==d[1:4:16][1],d[1:4:16]) && all(x->x==d[2:4:16][1],d[2:4:16]) && all(x->x==d[3:4:16][1],d[3:4:16]) && all(x->x==d[4:4:16][1],d[4:4:16])
is_class_2ij(d) = all(x->x==d[1:4][1],d[1:4]) && all(x->x==d[5:8][1],d[5:8]) && all(x->x==d[9:12][1],d[9:12]) && all(x->x==d[13:16][1],d[13:16])
is_class_2jk(d) = all(x->x==d[[1,2,9,10]][1],d[[1,2,9,10]]) && all(x->x==d[[3,4,11,12]][1],d[[3,4,11,12]]) && all(x->x==d[[5,6,13,14]][1],d[[5,6,13,14]]) && all(x->x==d[[7,8,15,16]][1],d[[7,8,15,16]])
is_class_2il(d) = all(x->x==d[[1,3,5,7]][1],d[[1,3,5,7]]) && all(x->x==d[[2,4,6,8]][1],d[[2,4,6,8]]) && all(x->x==d[[9,11,13,15]][1],d[[9,11,13,15]]) && all(x->x==d[[10,12,14,16]][1],d[[10,12,14,16]])
is_class_2jl(d) = all(x->x==d[[1,3,9,11]][1],d[[1,3,9,11]]) && all(x->x==d[[2,4,10,12]][1],d[[2,4,10,12]]) && all(x->x==d[[5,7,13,15]][1],d[[5,7,13,15]]) && all(x->x==d[[6,8,14,16]][1],d[[6,8,14,16]])
is_class_2ik(d) = all(x->x==d[[1,2,5,6]][1],d[[1,2,5,6]]) && all(x->x==d[[3,4,7,8]][1],d[[3,4,7,8]]) && all(x->x==d[[9,10,13,14]][1],d[[9,10,13,14]]) && all(x->x==d[[11,12,15,16]][1],d[[11,12,15,16]])
is_class_2(d) = is_class_2kl(d) || is_class_2ij(d) || is_class_2jk(d) || is_class_2il(d) || is_class_2jl(d) || is_class_2ik(d)
# 3rd order
is_class_3jkl(d) = (d[1] == d[9]) && (d[2] == d[10]) && (d[3] == d[11]) & (d[4] == d[12]) && (d[5] == d[13]) && (d[6] == d[14]) && (d[7] == d[15]) && (d[8] == d[16])
is_class_3ijk(d) = (d[1] == d[2]) && (d[3] == d[4]) && (d[5] == d[6]) & (d[7] == d[8]) && (d[9] == d[10]) && (d[11] == d[12]) && (d[13] == d[14]) && (d[15] == d[16])
is_class_3ikl(d) = (d[1] == d[5]) && (d[2] == d[6]) && (d[3] == d[7]) & (d[4] == d[8]) && (d[9] == d[13]) && (d[10] == d[14]) && (d[11] == d[15]) && (d[12] == d[16])
is_class_3ijl(d) = (d[1] == d[3]) && (d[2] == d[4]) && (d[5] == d[7]) & (d[6] == d[8]) && (d[9] == d[11]) && (d[10] == d[12]) && (d[13] == d[15]) && (d[14] == d[16])
is_class_3(d) = is_class_3jkl(d) || is_class_3ijk(d) || is_class_3ikl(d) || is_class_3ijl(d)
# 4th order
is_class_4(d) = !is_class_0(d) && !is_class_1(d) && !is_class_2(d) && !is_class_3(d)

# is norm d of a given class?
# 0th class
is_class_0_int(d_int) = is_class_0(d_int_to_bin(d_int))
# 1st class
is_class_1l_int(d_int) = is_class_1l(d_int_to_bin(d_int))
is_class_1k_int(d_int) = is_class_1k(d_int_to_bin(d_int))
is_class_1j_int(d_int) = is_class_1j(d_int_to_bin(d_int))
is_class_1i_int(d_int) = is_class_1i(d_int_to_bin(d_int))
is_class_1_int(d_int) = is_class_1(d_int_to_bin(d_int))  
# 2nd class
is_class_2kl_int(d_int) = is_class_2kl(d_int_to_bin(d_int))
is_class_2ij_int(d_int) = is_class_2ij(d_int_to_bin(d_int))
is_class_2jk_int(d_int) = is_class_2jk(d_int_to_bin(d_int))
is_class_2il_int(d_int) = is_class_2il(d_int_to_bin(d_int))
is_class_2jl_int(d_int) = is_class_2jl(d_int_to_bin(d_int))
is_class_2ik_int(d_int) = is_class_2ik(d_int_to_bin(d_int))
is_class_2_int(d_int) = is_class_2(d_int_to_bin(d_int))
# 3rd class
is_class_3jkl_int(d_int) = is_class_3jkl(d_int_to_bin(d_int)) # the subset of norms in Nakamura and Ohtsuki (2014)
is_class_3ijk_int(d_int) = is_class_3ijk(d_int_to_bin(d_int)) # the subset of norms in Ohtsuki and Iwasa (2004)
is_class_3ikl_int(d_int) = is_class_3ikl(d_int_to_bin(d_int))
is_class_3ijl_int(d_int) = is_class_3ijl(d_int_to_bin(d_int))
is_class_3_int(d_int) = is_class_3(d_int_to_bin(d_int))
# 4th order
is_class_4_int(d_int) = !is_class_0_int(d_int) && !is_class_1_int(d_int) && !is_class_2_int(d_int) && !is_class_3_int(d_int)

is_leading_eight_int(p_int,d_int) = (p_int == 11 && d_int == 50124) || (p_int == 11 && d_int == 53196) || (p_int == 10 && d_int == 50115) || (p_int == 10 && d_int == 50127) || (p_int == 10 && d_int == 53187) || (p_int == 10 && d_int == 53199) || (p_int == 10 && d_int == 50112) || (p_int == 10 && d_int == 53184)

"""
    is_L1_standing_int(p_int, d_int) -> Bool

Check if a moral system corresponds to the L1 Standing strategy.
L1 (Standing) is characterized by p_int = 11 and d_int = 53196.

The binary representation of d_int is:
1100 1111 1100 1100
"""
function is_L1_standing_int(p_int, d_int)
    return p_int == 11 && d_int == 53196
end

"""
    is_L2_consistent_standing_int(p_int, d_int) -> Bool

Check if a moral system corresponds to the L2 Consistent Standing strategy.
L2 (Consistent Standing) is characterized by p_int = 11 and d_int = 50124.

The binary representation of d_int is:
1100 0011 1100 1100
"""
function is_L2_consistent_standing_int(p_int, d_int)
    return p_int == 11 && d_int == 50124
end

"""
    is_L3_simple_standing_int(p_int, d_int) -> Bool

Check if a moral system corresponds to the L3 Simple Standing strategy.
L3 (Simple Standing) is characterized by p_int = 10 and d_int = 53199.

The binary representation of d_int is:
1100 1111 1100 1111
"""
function is_L3_simple_standing_int(p_int, d_int)
    return p_int == 10 && d_int == 53199
end

"""
    is_L4_SS_SJ_int(p_int, d_int) -> Bool

Check if a moral system corresponds to the L4 SS-SJ strategy.
L4 (SS-SJ) is characterized by p_int = 10 and d_int = 53187.

The binary representation of d_int is:
1100 1111 1100 0011
"""
function is_L4_SS_SJ_int(p_int, d_int)
    return p_int == 10 && d_int == 53187
end

"""
    is_L5_SJ_SS_int(p_int, d_int) -> Bool

Check if a moral system corresponds to the L5 SJ-SS strategy.
L5 (SJ-SS) is characterized by p_int = 10 and d_int = 50127.

The binary representation of d_int is:
1100 0011 1100 1111
"""
function is_L5_SJ_SS_int(p_int, d_int)
    return p_int == 10 && d_int == 50127
end

"""
    is_L6_stern_judging_int(p_int, d_int) -> Bool

Check if a moral system corresponds to the L6 Stern Judging strategy.
L6 (Stern Judging) is characterized by p_int = 10 and d_int = 50115.

The binary representation of d_int is:
1100 0011 1100 0011
"""
function is_L6_stern_judging_int(p_int, d_int)
    return p_int == 10 && d_int == 50115
end

"""
    is_L7_staying_int(p_int, d_int) -> Bool

Check if a moral system corresponds to the L7 Staying strategy.
L7 (Staying) is characterized by p_int = 10 and d_int = 53184.

The binary representation of d_int is:
1100 1111 1100 0000
"""
function is_L7_staying_int(p_int, d_int)
    return p_int == 10 && d_int == 53184
end

"""
    is_L8_judging_int(p_int, d_int) -> Bool

Check if a moral system corresponds to the L8 Judging strategy.
L8 (Judging) is characterized by p_int = 10 and d_int = 50112.

The binary representation of d_int is:
1100 0011 1100 0000
"""
function is_L8_judging_int(p_int, d_int)
    return p_int == 10 && d_int == 50112
end

"""
    get_leading_eight_name(p_int, d_int) -> String

Returns the name of the specific Leading Eight strategy if the moral system
matches one, or "Not Leading Eight" otherwise.
"""
function get_leading_eight_name(p_int, d_int)
    if is_L1_standing_int(p_int, d_int)
        return "L1 (Standing)"
    elseif is_L2_consistent_standing_int(p_int, d_int)
        return "L2 (Consistent Standing)"
    elseif is_L3_simple_standing_int(p_int, d_int)
        return "L3 (Simple Standing)"
    elseif is_L4_SS_SJ_int(p_int, d_int)
        return "L4 (SS-SJ)"
    elseif is_L5_SJ_SS_int(p_int, d_int)
        return "L5 (SJ-SS)"
    elseif is_L6_stern_judging_int(p_int, d_int)
        return "L6 (Stern Judging)"
    elseif is_L7_staying_int(p_int, d_int)
        return "L7 (Staying)"
    elseif is_L8_judging_int(p_int, d_int)
        return "L8 (Judging)"
    else
        return "Not Leading Eight"
    end
end

"""
    identify_leading_eight(df::DataFrame) -> DataFrame

Takes a DataFrame with p and d columns and adds columns for each Leading Eight strategy.
Also adds a column with the leading eight name for easy lookup.
Returns a new DataFrame with the additional columns.
"""
function identify_leading_eight(df::DataFrame)
    # Create a copy of the input DataFrame
    result = copy(df)
    
    # Add columns for each Leading Eight strategy
    result.is_L1_standing = [is_L1_standing_int(p, d) for (p, d) in zip(result.p, result.d)]
    result.is_L2_consistent_standing = [is_L2_consistent_standing_int(p, d) for (p, d) in zip(result.p, result.d)]
    result.is_L3_simple_standing = [is_L3_simple_standing_int(p, d) for (p, d) in zip(result.p, result.d)]
    result.is_L4_SS_SJ = [is_L4_SS_SJ_int(p, d) for (p, d) in zip(result.p, result.d)]
    result.is_L5_SJ_SS = [is_L5_SJ_SS_int(p, d) for (p, d) in zip(result.p, result.d)]
    result.is_L6_stern_judging = [is_L6_stern_judging_int(p, d) for (p, d) in zip(result.p, result.d)]
    result.is_L7_staying = [is_L7_staying_int(p, d) for (p, d) in zip(result.p, result.d)]
    result.is_L8_judging = [is_L8_judging_int(p, d) for (p, d) in zip(result.p, result.d)]
    
    # Add a column with the name of the Leading Eight strategy
    result.leading_eight_name = [get_leading_eight_name(p, d) for (p, d) in zip(result.p, result.d)]
    
    return result
end

function is_discriminating(p)
    pGB = getp(p, 1, 0)
    pGG = getp(p, 1,1)
    return pGB == 0 && pGG == 1
end

function is_discriminating_int(p_int)
    p = p_int_to_bin(p_int)
    return is_discriminating(p)
end

function is_consistent_int(p_int, d_int)
        p = p_int_to_bin(p_int) # convert action rule to binary array
        d = d_int_to_bin(d_int) # convert social norm to binary array
        pGG = getp(p,1,1)
		pBB = getp(p,0,0)
		pBG = getp(p,0,1)
        pGB = getp(p,1,0)
		dBB = getd(d,0,0,pBB,pBB)
		dBG = getd(d,0,1,pBG,pGB)
		dGB = getd(d,1,0,pGB,pBG)
		dGG = getd(d,1,1,pGG,pGG)
		dBBm = getd(d,0,0,1-pBB,pBB)
		dBGm = getd(d,0,1,1-pBG,pGB)
		dGBm = getd(d,1,0,1-pGB,pBG)
		dGGm = getd(d,1,1,1-pGG,pGG)
		if (dBB==1 && dBG==1 && dGB==1 && dGG==1 && dBBm==0 && dBGm==0 && dGBm==0 && dGGm==0)
			return true
		else
			return false
		end	
end

function is_consistent_discriminating_int(p_int,d_int)
	return is_consistent_int(p_int,d_int) && is_discriminating_int(p_int)
end

function is_conditional(p)
    pGG = getp(p,1,1)
    pBB = getp(p,0,0)
    pBG = getp(p,0,1)
    pGB = getp(p,1,0)
    if pGG == pBG == pGB == pBB
        return false
    else
        return true
    end
end


function is_self_approving_int(p_int, d_int)
    p = p_int_to_bin(p_int)
	d = d_int_to_bin(d_int)
	dij = [0, 0, 0, 0] # [dBB, dBG, dGB, dGG]
	for i in 0:1
		for j in 0:1
			dij[1+i*2+j] = getd(d,i,j,getp(p,i,j),getp(p,j,i))
		end
	end
    return dij[4] == 1 && sum([dij[1], dij[2], dij[3]]) >= 2
end

function is_self_disapproving_int(p_int, d_int)
    p = p_int_to_bin(p_int)
    d = d_int_to_bin_(d_int)
    dij = [0, 0, 0, 0] # [dBB, dBG, dGB, dGG]
	for i in 0:1
		for j in 0:1
			dij[1+i*2+j] = getd(d,i,j,getp(p,i,j),getp(p,j,i))
		end
	end
    return dij[4] == 0 && sum([dij[1], dij[2], dij[3]]) <= 1
end

function  is_conditional_self_cooperative_int(p_int, d_int)
    p = p_int_to_bin(p_int)
	pGG = getp(p,1,1)
	return (pGG == 1) && is_self_approving_int(p_int, d_int)
end


# we should double check this with a unit test.
function is_self_approving_mirror_int(p_int, d_int)
    p = p_int_to_bin(p_int)
	d = d_int_to_bin(d_int)
	dij = [0, 0, 0, 0] # [dBB, dBG, dGB, dGG]
	for i in 0:1
		for j in 0:1
			dij[1+i*2+j] = getd(d,i,j,getp(p,i,j),getp(p,j,i))
		end
	end
	if (dij[1] == 0) &&  (dij[2] == 0) && (dij[3] == 0) && (dij[4] == 1) 
		return true
	elseif (dij[1] == 0) &&  (dij[2] == 0) && (dij[3] == 1) && (dij[4] == 0) 
		return true
	elseif (dij[1] == 0) &&  (dij[2] == 1) && (dij[3] == 0) && (dij[4] == 0) 
		return true
	elseif (dij[1] == 0) &&  (dij[2] == 0) && (dij[3] == 0) && (dij[4] == 0) 
		return true
	else
		return false
	end

end


    

# Helper function for is_self_approving_mirror check
function is_self_approving_mirror(s::Strategy)
    dij = [getd(s,i÷2,i%2,getp(s,i÷2,i%2),getp(s,i%2,i÷2)) for i in 0:3]
    return (dij[1] == 0 && dij[2] == 0 && dij[3] == 0 && dij[4] == 1) ||
           (dij[1] == 0 && dij[2] == 0 && dij[3] == 1 && dij[4] == 0) ||
           (dij[1] == 0 && dij[2] == 1 && dij[3] == 0 && dij[4] == 0) ||
           (dij[1] == 0 && dij[2] == 0 && dij[3] == 0 && dij[4] == 0)
end

function  is_conditional_self_cooperative_mirror_int(p_int, d_int)
    p = p_int_to_bin(p_int)
	pBB = getp(p,0,0)
	return (pBB == 1) && is_self_approving_mirror_int(p_int, d_int)
end

function is_self_cooperative_int(p_int,d_int)
    return p_int == 15 || (is_conditional_self_cooperative_int(p_int,d_int) || is_conditional_self_cooperative_mirror_int(p_int,d_int))  
end

function compute_global_properties(df_local, χ, ϵ)
    # Filter dataframes by game type
    df_SH = filter(row -> row.game == "SH", df_local)
    df_SG = filter(row -> row.game == "SG", df_local)
    df_PD = filter(row -> row.game == "PD", df_local)

    # Create base dataframe with strategy information
    df_global = select(df_SH, [:p, :d])
    
    # Convert integer pairs to Strategy objects for computations
    strategies = [Strategy(p, d) for (p, d) in zip(df_global.p, df_global.d)]
    
    # Compute reputation equilibrium and cooperation index using Strategy objects
    df_global.resident_reputation_eq = [get_resident_reputation_eq(s, χ, ϵ) for s in strategies]
    
    # For cooperation index, we need to maintain compatibility with the original implementation
    # but use the new Strategy type internally
    df_global.cooperation_index = map(strategies) do s
        g = get_resident_reputation_eq(s, χ, ϵ)
        return (1-ϵ)*(g*(g*getp(s,1,1)+(1-g)*getp(s,1,0)) + 
                      (1-g)*(g*getp(s,0,1)+(1-g)*getp(s,0,0)))
    end
    
    # Add stability indices from the different games
    df_global.stability_index_PD = df_PD.stability_index
    df_global.stability_index_SG = df_SG.stability_index
    df_global.stability_index_SH = df_SH.stability_index

    # Compute weighted stability indices
    df_global.stability_index_uniform = (1/3)*(df_global.stability_index_PD + 
                                              df_global.stability_index_SG + 
                                              df_global.stability_index_SH)

    df_global.stability_index_area = (3/9)*(df_global.stability_index_PD) + 
                                    (2/9)*(df_global.stability_index_SG) + 
                                    (4/9)*(df_global.stability_index_SH)
    
    
    
    return df_global
end


function compute_local_properties(payoffs_filename, strategy_space, χ,ϵ)
    df_payoffs = CSV.read(payoffs_filename, DataFrame)
    games = unique(df_payoffs.game) 
    
    ##-------- Parameter checking -----------##
    @assert Set(games) ⊆ Set(GAME_TYPES) "There are undefined types of games"
    @assert !isempty(Set(games)) "The set of games to considered cannot be empty"
    @assert strategy_space ∈ Set(STRATEGY_SPACES) "Strategy space invalid"
    @assert !isempty(Set(strategy_space)) "A strategy space must be given"
    ##-------- End of parameter checking -----------##
    
    # Determine strategy space
    if strategy_space == "complete"
        strategies = MoralConsistency.complete_strategy_space()
    elseif strategy_space == "toy"
        strategies = MoralConsistency.toy_strategy_space()
    elseif strategy_space == "test"
        strategies = MoralConsistency.test_strategy_space()
    end
    
    # Create a thread-safe channel to collect results
    results_channel = Channel{DataFrame}(Threads.nthreads())
    
    # Split the strategy space into chunks for each thread
    n_strategies = length(strategies)
    chunk_size = ceil(Int, n_strategies / Threads.nthreads())
    
    # Pre-allocate the chunks vector
    n_chunks = Threads.nthreads()
    strategy_chunks = Vector{Vector{Tuple{Int, Int}}}(undef, n_chunks)
    
    # Fill the chunks vector
    for i in 1:n_chunks
        start_idx = (i-1) * chunk_size + 1
        end_idx = min(i * chunk_size, n_strategies)
        strategy_chunks[i] = strategies[start_idx:end_idx]
    end
    
    # Create a progress meter
    p = Progress(n_strategies; desc="Processing strategies: ")
    
    # Create an atomic counter for progress updates
    counter = Atomic{Int}(0)
    # Add mutex for progress updates
    progress_mutex = ReentrantLock()

    
    # Pre-allocate the results array
    all_results = Vector{DataFrame}(undef, length(strategy_chunks))

    # Process chunks in parallel using @threads
    Threads.@threads for i in eachindex(strategy_chunks)
        chunk = strategy_chunks[i]

        # Process the chunk
        chunk_results = MoralConsistency.stability_index(chunk, df_payoffs, χ, ϵ)
        
        # Store results in pre-allocated array
        all_results[i] = chunk_results

        # Update progress counter in batches to reduce contention
        local_count = length(chunk)
        new_count = atomic_add!(counter, local_count)
        
        # In the threaded section:
        if new_count % 50_000 == 0
            lock(progress_mutex) do
                update!(p, counter[])
            end
        end
    end
    
    # Combine all results into a single DataFrame
    df_results = vcat(all_results...)
    
    return df_results
end


function is_maximally_self_approving(p_int, d_int)

    # Convert d to binary form
    d_bin = d_int_to_bin(d_int)
        
    # Calculate shorthands
    shorthand_dGG = d_p_to_d_shorthand(d_bin, p_int, 1, 1)
    shorthand_dGB = d_p_to_d_shorthand(d_bin, p_int, 1, 0)
    shorthand_dBG = d_p_to_d_shorthand(d_bin, p_int, 0, 1)
    shorthand_dBB = d_p_to_d_shorthand(d_bin, p_int, 0, 0)
    if shorthand_dGG == 1 && shorthand_dGB == 1 && shorthand_dBG == 1 && shorthand_dBB == 1
        return true
    else
        return false
    end
end

# Helper function for d_p_to_d_shorthand
function getpshorthand(p::Int64, i::Int, j::Int)
    strat = digits(p, base=2, pad=4)
    return strat[PRECOMPUTED_P_INDICES[i+1, j+1]]
end

function d_p_to_d_shorthand(d::Vector{Int64}, p::Int64, r_one::Int, r_two::Int)::Int64
    action_one = getpshorthand(p, r_one, r_two)
    action_two = getpshorthand(p, r_two, r_one)
    return d[PRECOMPUTED_INDICES[r_one+1,r_two+1,action_one+1,action_two+1]]
end


function compute_universal_properties()

    all_space = MoralConsistency.complete_strategy_space()
    df_global = DataFrame(all_space, [:p, :d])

    # Convert integer pairs to Strategy objects for computations
    strategies = [Strategy(p, d) for (p, d) in zip(df_global.p, df_global.d)]
    
    df_global.is_conditional = map(strategies) do s
        return is_conditional(s)
    end

    # Compute boolean properties using Strategy objects
    df_global.is_self_approving = map(strategies) do s
        dij = [getd(s,i÷2,i%2,getp(s,i÷2,i%2),getp(s,i%2,i÷2)) for i in 0:3]
        return dij[4] == 1 && sum([dij[1], dij[2], dij[3]]) >= 2
    end
    df_global.is_self_disapproving = map(strategies) do s
        dij = [getd(s,i÷2,i%2,getp(s,i÷2,i%2),getp(s,i%2,i÷2)) for i in 0:3]
        return dij[4] == 0 && sum([dij[1], dij[2], dij[3]]) <= 1
    end
    df_global.is_self_cooperative = map(strategies) do s
        pGG = getp(s,1,1)
        pBB = getp(s,0,0)
        return s.p_int == 15 || 
               ((pGG == 1 && df_global.is_self_approving[findfirst(==(s), strategies)]) || 
                (pBB == 1 && is_self_approving_mirror(s)))
    end

    df_global.is_discriminating = map(strategies) do s
        return is_discriminating(s)
    end

    df_global.is_conditional_self_cooperative = map(strategies) do s
        return getp(s,1,1) == 1 && df_global.is_self_approving[findfirst(==(s), strategies)]
    end

    df_global.is_righteous .= df_global.is_conditional .& df_global.is_self_approving .& df_global.is_self_cooperative
    
    df_global.is_consistent = map(strategies) do s
        pGG = getp(s,1,1)
        pBB = getp(s,0,0)
        pBG = getp(s,0,1)
        pGB = getp(s,1,0)
        
        dBB = getd(s,0,0,pBB,pBB)
        dBG = getd(s,0,1,pBG,pGB)
        dGB = getd(s,1,0,pGB,pBG)
        dGG = getd(s,1,1,pGG,pGG)
        
        dBBm = getd(s,0,0,1-pBB,pBB)
        dBGm = getd(s,0,1,1-pBG,pGB)
        dGBm = getd(s,1,0,1-pGB,pBG)
        dGGm = getd(s,1,1,1-pGG,pGG)
        
        return dBB == 1 && dBG == 1 && dGB == 1 && dGG == 1 && 
               dBBm == 0 && dBGm == 0 && dGBm == 0 && dGGm == 0
    end
    
    
    df_global.is_consistent_discriminating = map(strategies) do s
        idx = findfirst(==(s), strategies)
        return df_global.is_consistent[idx] && df_global.is_discriminating[idx]
    end
    
    df_global.is_leading_eight = map(strategies) do s
        return (s.p_int == 11 && s.d_int == 50124) || 
               (s.p_int == 11 && s.d_int == 53196) || 
               (s.p_int == 10 && s.d_int == 50115) || 
               (s.p_int == 10 && s.d_int == 50127) || 
               (s.p_int == 10 && s.d_int == 53187) || 
               (s.p_int == 10 && s.d_int == 53199) || 
               (s.p_int == 10 && s.d_int == 50112) || 
               (s.p_int == 10 && s.d_int == 53184)
    end

    df_global.is_maximally_self_approving = map(strategies) do s
        return is_maximally_self_approving(s.p_int, s.d_int)
    end
    
    return df_global
end

    

end # module MoralConsistency
