# Minimal-data-collection-guarantee

## Prerequisites

numpy
cvxpy

## File structure


minimal-data-ldp/
        
        utils/
                ├── alphabet.py Create an alphabet for all attributes.
                ├── privacy_mechanism.py Superclass for all the mechanism
                        ├── randomized_response.py k-RR mechanism
                        ├── rappor_mechanism.py RAPPOR mechanism
                        ├── exponential_mechanism.py Exponential mechanism
                        ├── optimized_random_response.py Our solution
                                ├── convex_optimizer.py Convex optimizing code
                                ├── repetitive_optimizer.py Algorithm 1 in the paper
                ├── dataset_handler.py
                        ├── label_encoder.py
                ├── dataset_encoder.py
                ├── synthetic_dataset.py
                        ├── divergence.py
                ├── empirical_data.py
                ├── load_latent.py
                
                # Utility Functions
                ├── normalize_error_matrix.py Generate normalized error matrix
                ├── prior_distribution_calc.py Calculate prior distribution from a dataset
                ├── simpleinfotheory.py Information theoretic calculations
                ├── util_functions.py Necessary utility functions (Validating values etc.)
    
