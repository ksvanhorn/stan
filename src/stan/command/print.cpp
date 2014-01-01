#include <algorithm>
#include <iostream>
#include <iomanip>
#include <ios>
#include <stan/mcmc/chains.hpp>
#include <stan/command/print.hpp>

class printer_base0
{
protected:
  static stan::io::stan_csv get_stan_csv(std::string const & filename)
  {
    std::ifstream istrm(filename.c_str());
    return stan::io::stan_csv_reader::parse(istrm);
  }

  stan::mcmc::chains<> chains;
  Eigen::VectorXd warmup_times;
  Eigen::VectorXd sampling_times;  
  Eigen::VectorXi thin;
  std::string model_name;
  std::string algorithm;
  std::string engine;

  printer_base0(stan::io::stan_csv const & stan_csv0,
		std::vector<std::string> filenames)
    : chains(stan_csv0),
      warmup_times(filenames.size()),
      sampling_times(filenames.size()),
      thin(filenames.size()),
      model_name(stan_csv0.metadata.model),
      algorithm(stan_csv0.metadata.algorithm),
      engine(stan_csv0.metadata.engine)
  {
    warmup_times(0) = stan_csv0.timing.warmup;
    sampling_times(0) = stan_csv0.timing.sampling;
    thin(0) = stan_csv0.metadata.thin;
  }
};

class printer_base : protected printer_base0
{
protected:

  printer_base(std::vector<std::string> filenames)
    : printer_base0(get_stan_csv(filenames[0]), filenames)
  {
    typedef std::vector<std::string>::size_type size_type;
    for (size_type chain = 1; chain < filenames.size(); chain++)
    {
      stan::io::stan_csv stan_csv = get_stan_csv(filenames[chain]);
      chains.add(stan_csv);
      thin(chain) = stan_csv.metadata.thin;    
      warmup_times(chain) = stan_csv.timing.warmup;
      sampling_times(chain) = stan_csv.timing.sampling;
    }
  }
};

class printer : private printer_base
{
  static const int n = 9;
  static const int skip = 0;
  int sig_figs;
  Eigen::Matrix<std::string, Eigen::Dynamic, 1> headers;
  Eigen::MatrixXd values;

public:

  printer(int sig_figs_, std::vector<std::string> filenames)
    : printer_base(filenames),
      sig_figs(sig_figs_),
      headers(n),
      values(chains.num_params(), n)
  {

    // Prepare values
    values.setZero();
    Eigen::VectorXd probs(3);
    probs << 0.05, 0.5, 0.95;
    
    for (int i = 0; i < chains.num_params(); i++) {
      double sd = chains.sd(i);
      double n_eff = chains.effective_sample_size(i);
      values(i,0) = chains.mean(i);
      values(i,1) = sd / sqrt(n_eff);
      values(i,2) = sd;
      Eigen::VectorXd quantiles = chains.quantiles(i,probs);
      for (int j = 0; j < 3; j++)
        values(i,3+j) = quantiles(j);
      values(i,6) = n_eff;
      values(i,7) = n_eff / sampling_times.sum();
      values(i,8) = chains.split_potential_scale_reduction(i);
    }
  
    // Prepare header
    headers << 
      "Mean", "MCSE", "StdDev",
      "5%", "50%", "95%", 
      "N_Eff", "N_Eff/s", "R_hat";
  }

  void print_csv()
  {
    // Header output
    for (int i = 0; i < n; i++) {
      std::cout << ',' << headers(i);
    }
    std::cout << std::endl;
    
    // Value output
    for (int i = skip; i < chains.num_params(); i++) {
      if (!is_matrix(chains.param_name(i))) {
        std::cout << chains.param_name(i);
        for (int j = 0; j < n; j++) {
          std::cout.unsetf(std::ios::floatfield);
          std::cout << ','
		    << std::setprecision(sig_figs)
                    << values(i, j);
        }
        std::cout << std::endl;
      } else {
        std::vector<int> dims = dimensions(chains, i);
        std::vector<int> index(dims.size(), 1);
        int max = 1;
        for (std::size_t j = 0; j < dims.size(); j++)
          max *= dims[j];

        for (int k = 0; k < max; k++) {
          int param_index = i + matrix_index(index, dims);
          std::cout << chains.param_name(param_index);
          for (int j = 0; j < n; j++) {
            std::cout.unsetf(std::ios::floatfield);
            std::cout << ','
		      << std::setprecision(sig_figs)
		      << values(param_index, j);
          }
          std::cout << std::endl;
  	if (k < max - 1)
  	  next_index(index, dims);
        }
        i += max-1;
      }
    }
  }

  void print_initial_output()
  {
    double total_warmup_time = warmup_times.sum();
    double total_sampling_time = sampling_times.sum();
    std::cout << "Inference for Stan model: " << model_name << std::endl
	      << chains.num_chains() << " chains: each with iter=("
	      << chains.num_kept_samples(0);
    for (int chain = 1; chain < chains.num_chains(); chain++)
      std::cout << "," << chains.num_kept_samples(chain);
    std::cout << ")";
  
    // Timing output
    std::cout << "; warmup=(" << chains.warmup(0);
    for (int chain = 1; chain < chains.num_chains(); chain++)
      std::cout << "," << chains.warmup(chain);
    std::cout << ")";
                          
    std::cout << "; thin=(" << thin(0);
  
    for (int chain = 1; chain < chains.num_chains(); chain++)
      std::cout << "," << thin(chain);
    std::cout << ")";
                          
    std::cout << "; " << chains.num_samples() << " iterations saved."
	      << std::endl << std::endl;

    std::string warmup_unit = "seconds";
  
    if (total_warmup_time / 3600 > 1) {
      total_warmup_time /= 3600;
      warmup_unit = "hours";
    } else if (total_warmup_time / 60 > 1) {
      total_warmup_time /= 60;
      warmup_unit = "minutes";
    }
  
    int prec;
    prec = compute_precision(warmup_times(0), sig_figs, false);
    std::cout << "Warmup took ("
	      << std::fixed
	      << std::setprecision(prec)
	      << warmup_times(0);
    for (int chain = 1; chain < chains.num_chains(); chain++) {
      prec = compute_precision(warmup_times(chain), sig_figs, false);
      std::cout << ", " << std::fixed
		<< std::setprecision(prec)
		<< warmup_times(chain);
    }
    std::cout << ") seconds, ";
    prec = compute_precision(total_warmup_time, sig_figs, false);
    std::cout << std::fixed
	      << std::setprecision(prec)
	      << total_warmup_time << " " << warmup_unit << " total"
	      << std::endl;

    std::string sampling_unit = "seconds";
  
    if (total_sampling_time / 3600 > 1) {
      total_sampling_time /= 3600;
      sampling_unit = "hours";
    } else if (total_sampling_time / 60 > 1) {
      total_sampling_time /= 60;
      sampling_unit = "minutes";
    }

    prec = compute_precision(sampling_times(0), sig_figs, false);
    std::cout << "Sampling took ("
	      << std::fixed
	      << std::setprecision(prec)
	      << sampling_times(0);
    for (int chain = 1; chain < chains.num_chains(); chain++) {
      prec = compute_precision(sampling_times(chain), sig_figs, false);
      std::cout << ", " << std::fixed
		<< std::setprecision(prec)
		<< sampling_times(chain);
    }
    std::cout << ") seconds, ";
    prec = compute_precision(total_sampling_time, sig_figs, false);
    std::cout << std::fixed
	      << std::setprecision(prec)
	      << total_sampling_time << " " << sampling_unit << " total"
	      << std::endl;
    std::cout << std::endl;
  }

  void print_footer_output()
  {
    std::cout << std::endl;
    std::cout << "Samples were drawn using " << algorithm
	      << " with " << engine << "." << std::endl
	      << "For each parameter, N_Eff is a crude measure of effective sample size," << std::endl
	      << "and R_hat is the potential scale reduction factor on split chains (at " << std::endl
	      << "convergence, R_hat=1)." << std::endl
	      << std::endl;
  }

  std::size_t max_name_len()
  {
    // Compute largest variable name length
    size_t max_name_length = 0;
    for (int i = skip; i < chains.num_params(); i++) 
      if (chains.param_name(i).length() > max_name_length)
        max_name_length = chains.param_name(i).length();
    for (int i = 0; i < 2; i++) 
      if (chains.param_name(i).length() > max_name_length)
        max_name_length = chains.param_name(i).length();
    return max_name_length;
  }

  void print_text()
  {
    print_initial_output();

    // Compute largest variable name length
    size_t max_name_length = max_name_len();

    // Compute column widths
    Eigen::VectorXi column_widths(n);
    Eigen::Matrix<std::ios_base::fmtflags, Eigen::Dynamic, 1> formats(n);
    column_widths = calculate_column_widths(values, headers, sig_figs, formats);
  
    // Header output
    std::cout << std::setw(max_name_length + 1) << "";
    for (int i = 0; i < n; i++) {
      std::cout << std::setw(column_widths(i)) << headers(i);
    }
    std::cout << std::endl;
  
    // Value output
    for (int i = skip; i < chains.num_params(); i++) {
      if (!is_matrix(chains.param_name(i))) {
	std::cout << std::setw(max_name_length + 1) << std::left << chains.param_name(i);
	std::cout << std::right;
	for (int j = 0; j < n; j++) {
	  std::cout.setf(formats(j), std::ios::floatfield);
	  int prec = compute_precision(values(i,j), sig_figs, formats(j) == std::ios_base::scientific);
	  std::cout << std::setprecision(prec)
		    << std::setw(column_widths(j)) << values(i, j);
	}
	std::cout << std::endl;
      } else {
	std::vector<int> dims = dimensions(chains, i);
	std::vector<int> index(dims.size(), 1);
	int max = 1;
	for (std::size_t j = 0; j < dims.size(); j++)
	  max *= dims[j];

	for (int k = 0; k < max; k++) {
	  int param_index = i + matrix_index(index, dims);
	  std::cout << std::setw(max_name_length + 1) << std::left 
		    << chains.param_name(param_index);
	  std::cout << std::right;
	  for (int j = 0; j < n; j++) {
	    std::cout.setf(formats(j), std::ios::floatfield);
	    int prec = compute_precision(values(param_index,j), sig_figs, 
					 formats(j) == std::ios_base::scientific);
	    std::cout 
	      << std::setprecision(prec)
	      << std::setw(column_widths(j)) << values(param_index, j);
	  }
	  std::cout << std::endl;
	  if (k < max - 1)
	    next_index(index, dims);
	}
	i += max-1;
      }
    }

    print_footer_output();
  }

  void print_autocorr(int c)
  {
    size_t max_name_length = max_name_len();
    if (c < 0 || c >= chains.num_chains()) {
      std::cout << "Bad chain index " << c
		<< ", aborting autocorrelation display." << std::endl;
      return;
    }
    
    Eigen::MatrixXd autocorr(chains.num_params(), chains.num_samples(c));
    
    for (int i = 0; i < chains.num_params(); i++) {
      autocorr.row(i) = chains.autocorrelation(c, i);
    }
    
    // Format and print header
    std::cout << "Displaying the autocorrelations for chain " << c << ":"
	      << std::endl
	      << std::endl;
    
    const int n_autocorr = autocorr.row(0).size();
    
    int lag_width = 1;
    int number = n_autocorr; 
    while ( number != 0) { number /= 10; lag_width++; }

    std::cout << std::setw(lag_width > 4 ? lag_width : 4) << "Lag";
    for (int i = 0; i < chains.num_params(); ++i) {
      std::cout << std::setw(max_name_length + 1) << std::right
		<< chains.param_name(i);
    }
    std::cout << std::endl;

    // Print body  
    for (int n = 0; n < n_autocorr; ++n) {
      std::cout << std::setw(lag_width) << std::right << n;
      for (int i = 0; i < chains.num_params(); ++i) {
        std::cout << std::setw(max_name_length + 1) << std::right
		  << autocorr(i, n);
      }
      std::cout << std::endl;
    }
  } 

};

/**
 * The Stan print function.
 *
 * @param argc Number of arguments
 * @param argv Arguments
 * 
 * @return 0 for success, 
 *         non-zero otherwise
 */
int main(int argc, const char* argv[]) {
  
  if (argc == 1) {
    print_usage();
    return 0;
  }
  
  // Parse any arguments specifying filenames
  std::vector<std::string> filenames;
  bool csv_output = false;
  int corr_len = -1;
  int sig_figs = 2;
  
  for (int i = 1; i < argc; i++) {
    
    if (std::strncmp(argv[i], "--autocorr=", 11) == 0) {
      corr_len = atoi(argv[i] + 11);
      continue;
    }
    
    if (std::strncmp(argv[i], "--sig_figs=", 11) == 0) {
      sig_figs = atoi(argv[i] + 11);
      continue;
    }

    if (std::strcmp(argv[i], "--csv_output") == 0) {
      csv_output = true;
      continue;
    }
    
    if (std::string("--help") == std::string(argv[i])) {
      print_usage();
      return 0;
    }
    
    std::ifstream ifstream(argv[i]);
    if (ifstream.good())
      filenames.push_back(argv[i]);
    else
      std::cerr << "File " << argv[i] << " not found" << std::endl;
  }
  
  if (!filenames.size()) {
    std::cerr << "No valid input files, exiting." << std::endl;
    return 0;
  }
  
  printer prnt(sig_figs, filenames);
  
  if (csv_output)
    prnt.print_csv();
  else
    prnt.print_text();
  
  // Print autocorrelation, if desired
  if (!csv_output && corr_len >= 0)
    prnt.print_autocorr(corr_len);
      
  return 0;
        
}
