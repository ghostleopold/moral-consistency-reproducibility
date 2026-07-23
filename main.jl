using Pkg
Pkg.activate(".")
using Revise
using CSV
using DataFrames
using JSON
using ArgParse
using MoralConsistency

# Main function
function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--json"
            help = "The configuration json"
    end
    return parse_args(s)
end

filepath = nothing

parsed_args = parse_commandline()
if haskey(parsed_args, "json") && (parsed_args["json"] !== nothing)
    filepath = parsed_args["json"]
else
    #use testing file
    filepath = "toy_input.json"
    # filepath = "complete_chi_0.01_epsilon_0.01.json"
    println("Running test file...")
end


##------- Paramters ---##

params_dict = JSON.parsefile(filepath)


χ = params_dict["chi"]
ϵ = params_dict["epsilon"]
payoffs_filename = params_dict["payoffs_filename"]
strategy_space = params_dict["strategy_space"]
output_filename_local = params_dict["output_filename_local"]
output_filename_global = params_dict["output_filename_global"]

# Initialize thread caches at runtime
MoralConsistency.initialize_thread_caches()

##------- End of parameters ---##
df_local = MoralConsistency.compute_local_properties(payoffs_filename, strategy_space, χ,ϵ)
CSV.write(output_filename_local, df_local)
df_global = MoralConsistency.compute_global_properties(df_local, χ,ϵ)
CSV.write(output_filename_global, df_global)


# df_universal = MoralConsistency.compute_universal_properties()
# CSV.write("universal_properties.csv", df_universal)