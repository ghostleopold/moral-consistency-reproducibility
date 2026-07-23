using Pkg
Pkg.activate(".")
using DataFrames, CSV, ParetoFront

# Function to calculate Pareto frontier and add column to dataframe
function add_pareto_frontier!(df)
    # Extract points for Pareto calculation
    points = Matrix(df[:, [:stability_index_uniform, :cooperation_index]])
    
    # Initialize pareto set and settings
    pareto_set = Set{Vector{Float64}}()
    min_idxs = []  # Columns to minimize (none in this case)
    max_idxs = [1, 2]  # Columns to maximize (both in this case)
    
    # Calculate Pareto frontier
    for i in 1:size(points, 1)
        update_pareto!(pareto_set, points[i, :], min_idxs, max_idxs)
    end
    
    # Convert to tuples for efficient lookup
    pareto_tuples = Set(Tuple(p) for p in collect(pareto_set))
    
    # Add the pareto_front column to the dataframe
    df[!, :pareto_front] = [Tuple([row.stability_index_uniform, row.cooperation_index]) in pareto_tuples 
                          for row in eachrow(df)]
    
    return df
end

# Main processing function
function process_csv_files()
    # Path to directory
    dir_path = "data/optimised_code_19.02.2025/"
    # dir_path = "data/test/"
    
    # Get all CSV files matching the pattern
    files = filter(file -> occursin(r"global_complete_chi_0\.\d+_epsilon_0\.\d+\.csv", file), 
                  readdir(dir_path, join=true))
    
    # Process each file
    for file in files
        println("Processing: $file")
        
        # Read CSV to dataframe
        df = CSV.read(file, DataFrame)
        
        # Add pareto frontier column
        add_pareto_frontier!(df)
        
        # Save back to the same file (overwriting the original)
        CSV.write(file, df)
        
        println("Completed: $file")
    end
    
    println("All files processed!")
end

# Run the main function
process_csv_files()