from NeuroCharter import Study
import time
print time.time

# Study('DataNT.csv', 'cross validation', num_outputs=4, data_partition=(75, 15),
#        tolerance=0.0000001, learning_rate=0.4, maximum_epochs=100,
#        adapt_learning_rate=False, annealing_value=2000,
#        display_graph_windows=False, display_graph_pdf=False,
#        data_file_has_titles=True, data_file_has_brief_titles=True,
#        minimum_slope_to_consider_overfitting=2, number_of_epochs_for_overfit_check=10,
#        relative_importance_method="M")


# Querying for inputs: Temperature,Relative Humidity, Radiation, Site
Study(purpose= "advanced query", previous_study_data_file="NrCh_StoredANN_B44H.nsr", start_time=time.time(),
     input_parameters_suggested_values= ([10, 45, 5], 7.5, (0.5, 0.8), ('A', 'C')))
