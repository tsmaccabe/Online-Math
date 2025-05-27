using Distributions, Images, FiniteDifferences, FunctionZeros, Roots, SpecialFunctions, Plots, QuadGK, Cubature

include("quantum_number_combos.jl")

iszero(x, eps=1e-50) = abs(x) < eps

W(l::Real, k_nl::Real, r0::Real, a::Real) = besselj(l, k_nl*r0)*bessely(l, k_nl*a) - besselj(l, k_nl*a)*bessely(l, k_nl*r0)

A(l, k_nl, a, W_nl) = bessely(l, k_nl*a) / W_nl
B(l, k_nl, a, W_nl) = -besselj(l, k_nl*a) / W_nl

function enclose_R(l, k_nl, r0, a)
    W_nl = W(l, k_nl, r0, a)

    return (r) -> (besselj(l, k_nl*r)*bessely(l, k_nl*a) - besselj(l, k_nl*a)*bessely(l, k_nl*r))/W_nl
end

function k(n, l, a, b, k_max = 100)
    Q(k) = besselj(l, k*a)*bessely(l, k*b) - besselj(l, k*b)*bessely(l, k*a)
    scaled_zeros_list = Float64[]
    k_min = 0.
    while length(scaled_zeros_list) < n
        k_max *= 2
        append!(scaled_zeros_list, fzeros((k) -> Q(k), k_min, k_max))
        k_min = copy(k_max)
    end
    return scaled_zeros_list[n]
end

F(φ, l) = exp(1im*l*φ)

function γ(R_nl, a, b; rtol=1e-8)
    integrand = (r) -> r*R_nl(r)
    result, _ = quadgk(integrand, b, a; rtol=rtol)
    result = (2π*result)^(-1)
    return result
end

function γ_indefinite(R_nl, r, n, l, r0, a, b)
    k_nl_minus1 = k(n, abs(l) - 1, a, b)
    R_nl_minus1 = enclose_R(abs(l) - 1, k_nl_minus1, r0, a)
    k_nl_plus1 = k(n, abs(l) + 1, a, b)
    R_nl_plus1 = enclose_R(abs(l) + 1, k_nl_plus1, r0, a)

    return (r)*((π/2)*abs(R_nl(r)^2 - R_nl_minus1(r)*R_nl_plus1(r)))^(-0.5)
end
γ_analytic(R_nl, n, l, r0, a, b) = γ_indefinite(R_nl, a, n, l, r0, a, b) - γ_indefinite(R_nl, b, n, l, r0, a, b)

ζ(r, φ, R_nl, F_l, γ_nl) = γ_nl*R_nl(r)*F_l(φ)
function ψ(x, y, R_nl, F_l, a, b)
    r = sqrt(x^2 + y^2)
    if (r < b) | (r > a)
        return 0.
    else
        φ = atan(y, x)
        return R_nl(r)*F_l(φ)
    end
end

function γnl_abs(l::Real, k::Real, r0::Real, a::Real, b::Real)
    l = abs(l)
    Wnl = W(l, k, r0, b)
    return (Wnl/sqrt(2π)) * (bessely(l, k*b)^2 * I1(l, k, a, b) + besselj(l, k*b)^2 * I2(l, k, a, b) - 2bessely(l, k*b)*besselj(l, k*b) * I3(l, k, a, b))^(-0.5)
end

function energy_integral(R_nl, l::Real, Bnl::Real, k::Real, a::Real, b::Real; rtol::Real=1e-8)
    ab = (a + b)/2
    l = abs(l)

    Wnl = W(l, k, ab, a)

    A = ab*bessely(l, k*a)/Wnl
    B = -ab*besselj(l, k*a)/Wnl

    return (2π*Bnl^2) * result
end

function polar_integral(f_grid, r_rng, s_rng)
    quantity = 0.
    for (ir, r) in enumerate(r_rng)
        for is in axes(s_rng)[1]
            quantity += r*f_grid[ir, is]
        end
    end
    return quantity/length(r_rng)
end

function cart_integral(f_grid, x_rng, y_rng)
    quantity = 0.
    for ix in axes(x_rng)[1]
        for iy in axes(y_rng)[1]
            quantity += f_grid[ix, iy]
        end
    end
    return quantity/length(x_rng)
end

function besselr(n, r_rng)
    return besselj.(n, besselj_zero(n, n)*r_rng)
end

#radial(r, a, b, n, l) = sin(π*n*(r-a)/(b-a))*(r-a)^l
#radial(a, b, n, l) = #*(r-a)^abs(l)
angular(s, l) = exp(1im*l*s)

trial_wavefunction(fr, r, s, l) = fr(r)*angular(s, l)
function trial_wavefunction_cart(fr, x, y, a, b, l)
    r = sqrt(x^2 + y^2)
    s = atan(y, x)
    if (r > a) | (r < b)
        return 0+0im
    else
        return trial_wavefunction(fr, r, s, l) 
    end
end

b, a = 1., 4.
ab = (a + b)/2

const_t0 = 1e-2

dx = 0.1
x_rng = -4:dx:4
x_num = length(x_rng)
dy = 0.1
y_rng = -4:dy:4
y_num = length(y_rng)

energy_basis = Float64[]
energy_osc = Function[]
wave_bases = Matrix{Complex}[]
momentum_exp = Tuple{ComplexF64, ComplexF64}[]
N1, N2 = 1, 4
nl_combos = sort([((n, -n) for n in N1:N2)..., ((n, l) for n in N1:2:N2 for l in floor(Int, n/2):n)...])
#nl_combos =  [(8, -4), (9, 1), (10, 2), (11, 3)]
for (inl, (n, l)) in enumerate(nl_combos)
    println(n,", ", l)

    k_nl = k(n, abs(l), a, b)
    R_nl = enclose_R(abs(l), k_nl, ab, a)
    γ_nl = γ(R_nl, a, b)
    println(γ_nl)

    F_l = (φ) -> F(φ, l)

    ψ_nl(x, y) = ψ(x, y, R_nl, F_l, a, b)
    
    basis_nl = reshape([ψ_nl(x, y) for x in x_rng for y in y_rng], (x_num, y_num))*dx*dy

    basis_density = abs.(basis_nl)^2
    γ_nl_sq_inv = sum(basis_density)
    basis_density /= γ_nl_sq_inv

    push!(wave_bases, basis_nl)
    push!(energy_basis, 0.)
end

k_11 = k(1, 1, a, b)
trial_radial_g, kg = enclose_R(1, k_11, ab, a)
Emin = abs(minimum(energy_basis))
Eg = min(Emin, abs(energy_integral(trial_radial_g, 1, Bnl_abs(1, kg, ab, a, b), kg, a, b)))
energy_basis .+= Eg + 0.
for (inl, (n, l)) in enumerate(nl_combos)
    push!(energy_osc, (t) -> exp(1im * energy_basis[inl] * t))
end

if length(nl_combos) == 1
    states_re_rng = 1
    states_im_rng = 1
else
    states_re_rng = [cos(rand_num) for rand_num in rand(Uniform(0, 2π), size(nl_combos))]#range(0, 1, length(nl_combos))
    states_im_rng = [sin(rand_num) for rand_num in rand(Uniform(0, 2π), size(nl_combos))]#rand(Uniform(-1., 1.), size(nl_combos))
end

state_coeffs = [rp - 1im*ip for (rp, ip) in zip(states_re_rng, states_im_rng)]
#state_coeffs /= sqrt(sum(conj(state_coeffs).*state_coeffs))

τ = 1/mean(energy_basis)

max_period = 2π/minimum(energy_basis)
t_rng = 0.:0.0006τ:2τ
rgb_list = []
for (it, t) in enumerate(t_rng)
    println(it/length(t_rng))

    state_coeffs .*= [f(t) for f in energy_osc]
    wave = sum(state_coeffs .* wave_bases)
    trial_values = wave_bases[1]

    density = real(conj(wave) .* wave)
    density_sum = cart_integral(density, x_rng, y_rng)

    density ./= density_sum
    #trial_values ./= sqrt(density_sum)
    
    for ix in axes(density)[1]
        for iy in axes(density)[1]
            if iszero(density[ix, iy])
                density[ix, iy] = 0.
            end
        end
    end

    red_amp = 0*density#zeros(size(density))#imag(wave)
    green_amp = log.(1. .+ 3*density)
    blue_amp = log.(1. .+ 3*density)#zeros(size(density))#real(wave)

    rgb_amp = Array{Float64}(undef, 3, x_num, y_num)
    for ix in axes(wave)[1]
        for iy in axes(wave)[1]
            rgb_amp[:, ix, iy] = [red_amp[ix, iy], green_amp[ix, iy], blue_amp[ix, iy]]
        end
    end

    push!(rgb_list, rgb_amp)
end

rgb_min = minimum(minimum(rgb) for rgb in rgb_list)
rgb_max = maximum(maximum(rgb) for rgb in rgb_list)

for (i, rgb_amp) in enumerate(rgb_list)
    rgb_amp = (rgb_amp .- rgb_min)./(rgb_max .- rgb_min)

    rgb_list[i] = rgb_amp
end

anim = @animate for rgb_amp in rgb_list    
    for (ix, x) in enumerate(x_rng)
        for (iy, y) in enumerate(y_rng)
            if (sqrt(x^2 + y^2) < a) | (sqrt(x^2 + y^2) > b)
                rgb_amp[:, ix, iy] = 0.2*[1., 1., 1.]
            end
        end
    end

    plot(Images.colorview(RGB, rgb_amp))
    #plot(Images.colorview(Gray, density))
end

gif(anim, "annular.mp4", fps = 30)

#=
wave_d1 = diff(diff(wave; dims=1); dims=2)
wave_p = diff(diff(wave_d1; dims=1); dims=2)

red_amp = imag(wave_d1)
green_amp = real(wave_d1)
blue_amp = zeros(size(wave_d1))

rgb_amp = Array{Float64}(undef, 3, x_num, y_num)
for ix in axes(wave_d1)[1]
    for iy in axes(wave_d1)[2]
        rgb_amp[:, ix, iy] = [red_amp[ix, iy], green_amp[ix, iy], blue_amp[ix, iy]]
    end
end
rgb_amp = (rgb_amp .- minimum(rgb_amp))/(maximum(rgb_amp) - minimum(rgb_amp))

for ix in axes(trial_values)[1]
    for iy in axes(trial_values)[1]
        if iszero(density[ix, iy])
            rgb_amp[:, ix, iy] = [0.2, 0.2, 0.2]
        end
    end
end

ps9 = Images.colorview(RGB, rgb_amp)
ps10 = Images.colorview(Gray, density./maximum(density))

display(ps9)
display(ps10)

save("wf.png", ps9)
save("df.png", ps10)
=#