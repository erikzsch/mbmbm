library(ggplot2)
library(patchwork)

##Replace with with the relevant paths for your system
basic_folder = ""
target_folder = ""


create_averaged_ranks = function(ranked_data){
  averaged_ranks = aggregate(ranked ~ model + selector + params, ranked_data, mean)
  quartile_ranks = aggregate(ranked ~ model + selector + params, ranked_data, function(x) quantile(x))
  
  averaged_ranks = cbind(averaged_ranks, as.data.frame(quartile_ranks[["ranked"]]))
  colnames(averaged_ranks)[5:9] = c("min", "low.quart", "median", "upper.quart", "max")
  rm(quartile_ranks, ranked_data)
  
  averaged_ranks$selector = gsub("GenericUnivariateSelect", "GUS", averaged_ranks$selector)
  averaged_ranks$pasted = paste(averaged_ranks$selector,
                                sapply(strsplit(averaged_ranks$params, "score_func::"), function(x) {ifelse(length(x) == 2, x[2], "")} ))
  averaged_ranks$pasted_full = trimws(paste(averaged_ranks$model, averaged_ranks$selector,
                                     sapply(strsplit(averaged_ranks$params, "score_func::"), function(x) {ifelse(length(x) == 2, x[2], "")} )))
  
  averaged_ranks$pasted_full_factor = factor(averaged_ranks$pasted_full,
                                             levels = averaged_ranks$pasted_full[order(averaged_ranks$ranked)])
  
  return (averaged_ranks)
}

create_average_time_ranks = function(time_data){
  time_data$full_selector = paste(time_data$model, time_data$selector, time_data$params, sep = " -- ")
  aggregated_time = aggregate(total ~ dataset + model + selector + params, time_data, mean)
  
  ###set up ranks
  counter = 1
  for (dataset in unique(aggregated_time$dataset)){
    subset_result_data = aggregated_time[aggregated_time$dataset == dataset, ]
    subset_result_data$ranked = rank(subset_result_data$total)
    
    if(counter == 1){
      collected_ranked_time = subset_result_data
    }else{
      collected_ranked_time = rbind(collected_ranked_time, subset_result_data) 
    }
    counter = counter + 1
  }
  rm(aggregated_time, subset_result_data, time_data, counter, dataset)
  
  averaged_time = aggregate(ranked ~ model + selector + params, collected_ranked_time, mean)
  quartile_time = aggregate(ranked ~ model + selector + params, collected_ranked_time, function(x) quantile(x))
  averaged_time = cbind(averaged_time, as.data.frame(quartile_time[["ranked"]]))
  colnames(averaged_time)[5:9] = c("min", "low.quart", "median", "upper.quart", "max")

  averaged_time$selector = gsub("GenericUnivariateSelect", "GUS", averaged_time$selector)
  averaged_time$pasted = paste(averaged_time$selector,
                               sapply(strsplit(averaged_time$params, "score_func::"), function(x) {ifelse(length(x) == 2, x[2], "")} ))
  averaged_time$pasted_full = paste(averaged_time$model, averaged_time$selector,
                                    sapply(strsplit(averaged_time$params, "score_func::"), function(x) {ifelse(length(x) == 2, x[2], "")} ))
  averaged_time$pasted_full_factor = factor(averaged_time$pasted_full, 
                                            levels = averaged_time$pasted_full[order(averaged_time$ranked)])
  return(averaged_time)
}

prepare_rank_significance = function(input_file){
  rank_significance = read.csv(input_file, row.names = 1)
  rank_significance$padj = p.adjust(rank_significance$V7, method="fdr") < 0.05
  rank_significance$V2 = gsub("GenericUnivariateSelect", "GUS", rank_significance$V2)
  rank_significance$V3 = sapply(strsplit(rank_significance$V3, "score_func::"), function(x) {ifelse(length(x) == 2, x[2], "")} )
  rank_significance$V5 = gsub("GenericUnivariateSelect", "GUS", rank_significance$V5)
  rank_significance$V6 = sapply(strsplit(rank_significance$V6, "score_func::"), function(x) {ifelse(length(x) == 2, x[2], "")} )
  rank_significance$pasted_1 = trimws(paste(rank_significance$V1, rank_significance$V2, rank_significance$V3))
  rank_significance$pasted_2 = trimws(paste(rank_significance$V4, rank_significance$V5, rank_significance$V6))
  return(rank_significance)
}

create_line_plot = function(averaged_ranks, ylab_label, legend = TRUE) {
  fig_ranked_lines = ggplot(averaged_ranks, aes(x = pasted_full_factor, color = model, 
                                                y = low.quart, yend = upper.quart, label = pasted)) + 
    geom_segment() + 
    geom_point(aes(y = ranked)) + 
    theme_bw() + 
    ylab(ylab_label) +  # Set Y-axis label
    xlab("")            # Set X-axis label to blank
  
  if (legend) {
    fig_ranked_lines = fig_ranked_lines + 
      scale_color_discrete(name = "ML model") +
      theme(
        axis.title.x = element_blank(),                        # Remove X-axis title
        axis.title.y = element_text(size = 12, face = "bold"), # Bold and larger Y-axis label
        axis.text.y = element_text(size = 10, face = "bold", color = "black"), # Bold and larger Y-axis tick values
        axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5,      # Bold and larger X-axis text
                                   size = 14, face = "bold", color = "black"),
        legend.title = element_text(size = 14, face = "bold"), # Bold and larger legend title
        legend.text = element_text(size = 14, face = "bold"),  # Bold and larger legend text
        panel.grid.major = element_blank(), 
        panel.grid.minor = element_blank(),
        axis.line.y = element_line(color = "black")
      )
  } else {
    fig_ranked_lines = fig_ranked_lines + 
      theme(
        axis.title.x = element_blank(),                        # Remove X-axis title
        axis.title.y = element_text(size = 12, face = "bold"), # Bold and larger Y-axis label
        axis.text.y = element_text(size = 10, face = "bold", color = "black"), # Bold and larger Y-axis tick values
        axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5,      # Bold and larger X-axis text
                                   size = 14, face = "bold", color = "black"),
        legend.position = "none",                              # Remove legend
        panel.grid.major = element_blank(), 
        panel.grid.minor = element_blank(),
        axis.line.y = element_line(color = "black")
      )
  }
  return(fig_ranked_lines)
}



ranked_data_regression = read.csv(paste0(basic_folder, "results_regression/all_ranks.csv"), row.names=1)
averaged_ranks_regression = create_averaged_ranks(ranked_data_regression)
rm(ranked_data_regression)
  
time_data_regression = read.csv(paste0(basic_folder, "results_regression/collected.csv"), row.names=1)
averaged_time_regression = create_average_time_ranks(time_data_regression)
rm(time_data_regression)

ranked_data_regression_frac = read.csv(paste0(basic_folder, "results_regression_fraction/all_ranks.csv"), row.names=1)
averaged_ranks_regression_frac = create_averaged_ranks(ranked_data_regression_frac)
rm(ranked_data_regression_frac)

time_data_regression_frac = read.csv(paste0(basic_folder, "results_regression_fraction/collected.csv"), row.names=1)
averaged_time_regression_frac = create_average_time_ranks(time_data_regression_frac)
rm(time_data_regression_frac)

ranked_data_classification = read.csv(paste0(basic_folder, "results_classif/all_ranks.csv"), row.names=1)
averaged_ranks_classification = create_averaged_ranks(ranked_data_classification)
rm(ranked_data_classification)

time_data_classification = read.csv(paste0(basic_folder, "results_classif/collected.csv"), row.names=1)
averaged_time_classification = create_average_time_ranks(time_data_classification)
rm(time_data_classification)


lines_regression = create_line_plot(averaged_ranks_regression, "Rank of performance (R²)", F)
times_regression = create_line_plot(averaged_time_regression, "Rank of total runtime", T)
lines_regression_rel = create_line_plot(averaged_ranks_regression_frac, "Rank of performance (R²)", F)
times_regression_rel = create_line_plot(averaged_time_regression_frac, "Rank of total runtime", T)
lines_classification = create_line_plot(averaged_ranks_classification, "Rank of performance (F1)", F)
times_classification = create_line_plot(averaged_time_classification, "Rank of total runtime", T)

benchmark_results_ranked_ordered = (lines_regression + times_regression) / 
  (lines_regression_rel + times_regression_rel) / 
  (lines_classification + times_classification)

# Customize tags for large and bold labels
final_plot = benchmark_results_ranked_ordered + plot_annotation(
  tag_levels = 'A', # Generate tags A, B, C, ...
  theme = theme(
    plot.tag = element_text(size = 20, face = "bold") # Bold and large tags
  )
)

# Save the final plot
ggsave(
  filename = paste0(target_folder, "ranked_approaches_all.png"),
  plot = final_plot,
  width = 18,
  height = 21
)

#####plot heatmaps

create_significance_heatmap = function(rank_significance, averaged_ranks, legend = TRUE) {
  heatmap_performance = ggplot(rank_significance, 
                               aes(
                                 reorder(pasted_1, averaged_ranks[pmatch(pasted_1, averaged_ranks$pasted_full, duplicates.ok = T), "ranked"]), 
                                 reorder(pasted_2, averaged_ranks[pmatch(pasted_2, averaged_ranks$pasted_full, duplicates.ok = T), "ranked"]), 
                                 fill = padj
                               )) + 
    geom_tile() + 
    theme_bw() + 
    xlab("") + 
    ylab("") + 
    theme(
      # Common axis text and titles
      axis.title.x = element_text(size = 14, face = "bold"),  # Bold and larger X-axis title
      axis.title.y = element_text(size = 14, face = "bold"),  # Bold and larger Y-axis title
      axis.text.x = element_text(angle = 90, vjust = 0.4, hjust = 1, size = 14, face = "bold"),  # Bold X-axis text
      axis.text.y = element_text(size = 14, face = "bold"),  # Bold Y-axis text
      plot.title = element_text(size = 16, face = "bold")  # Bold plot title
    )
  
  if (legend) {
    heatmap_performance = heatmap_performance + 
      scale_fill_grey(name = "Significantly\ndifferent?") +
      theme(
        legend.title = element_text(size = 13, face = "bold"),  # Bold legend title
        legend.text = element_text(size = 13, face = "bold")   # Bold legend text
      )
  } else {
    heatmap_performance = heatmap_performance + 
      theme(
        legend.position = "none",  # Remove legend
        axis.text.x = element_text(angle = 90, vjust = 0.4, hjust = 1, size = 14, face = "bold"),  # Bold X-axis text
        axis.text.y = element_text(size = 14, face = "bold")  # Bold Y-axis text
      ) + 
      scale_y_discrete(position = "right") +
      scale_fill_grey()
  }
  
  return(heatmap_performance)
}
  
rank_significance_regression = prepare_rank_significance(paste0(basic_folder, "results_regression/ranked_approaches_significance.csv"))
rank_significance_time_regression = prepare_rank_significance(paste0(basic_folder, "results_regression/ranked_approaches_significance_time.csv"))
rank_significance_regression_relative = prepare_rank_significance(paste0(basic_folder, "results_regression_fraction/ranked_approaches_significance.csv"))
rank_significance_time_regression_relative = prepare_rank_significance(paste0(basic_folder, "results_regression_fraction/ranked_approaches_significance_time.csv"))
rank_significance_classification = prepare_rank_significance(paste0(basic_folder, "results_classif/ranked_approaches_significance.csv"))
rank_significance_time_classificaiton = prepare_rank_significance(paste0(basic_folder, "results_classif/ranked_approaches_significance_time.csv"))


heat_performance_regression = create_significance_heatmap(rank_significance_regression, averaged_ranks_regression, F)
heat_time_regression = create_significance_heatmap(rank_significance_time_regression, averaged_time_regression, T)
heat_performance_regression_relative = create_significance_heatmap(rank_significance_regression_relative, averaged_ranks_regression_frac, F)
heat_time_regression_relative = create_significance_heatmap(rank_significance_time_regression_relative, averaged_time_regression_frac, T)
heat_performance_classification = create_significance_heatmap(rank_significance_classification, averaged_ranks_classification, F)
heat_time_classification = create_significance_heatmap(rank_significance_time_classificaiton, averaged_time_classification, T)


ff = (heat_performance_regression + heat_time_regression)/
  (heat_performance_regression_relative + heat_time_regression_relative)/
  (heat_performance_classification+heat_time_classification)

ggsave(paste0(target_folder, "significance_approaches_all.png"), 
       ff + plot_annotation(tag_levels = 'A'),
       width = 25, height = 34)

#############
###visualize at the differences between the RF None and the other RF approaches

in_file = paste0(basic_folder, "results_regression/collected.csv")
result_data = read.csv(in_file, row.names = 1)
result_data$full_selector = paste(result_data$model, result_data$selector, result_data$params, sep = " -- ")
aggregated_data = aggregate(R2Score ~ dataset + model + selector + params, result_data, mean)

counter = 1
for (this_dataset in unique(aggregated_data$dataset)){
  aggregated_subset = aggregated_data[aggregated_data$dataset == this_dataset, ]
  relevant_RF_result = aggregated_subset[(aggregated_subset$model == "RandomForestRegressor") & (aggregated_subset$selector == "None"), "R2Score"]
  
  aggregated_subset$R2Score_rel = aggregated_subset$R2Score - relevant_RF_result
  
  if (counter == 1){
    collected_aggregated_data = aggregated_subset
  }else{
    collected_aggregated_data = rbind(collected_aggregated_data, aggregated_subset)
  }
  counter = counter + 1
}

collected_aggregated_data = collected_aggregated_data[collected_aggregated_data$selector != "None" | collected_aggregated_data$model != "RandomForestRegressor", ]

collected_aggregated_data$selector = gsub("GenericUnivariateSelect", "GUS", collected_aggregated_data$selector)
collected_aggregated_data$selector = gsub("FastCorrelationBasedFilter", "FCBF", collected_aggregated_data$selector)

collected_aggregated_data$pasted = paste(collected_aggregated_data$selector,
                                         sapply(strsplit(collected_aggregated_data$params, "score_func::"), function(x) {ifelse(length(x) == 2, x[2], "")} ))

aggregated_data_rf_subset = aggregated_data[(aggregated_data$model == "RandomForestRegressor") & (aggregated_data$selector == "None"), ]
collected_aggregated_data$dataset = sapply(collected_aggregated_data$dataset, function(x) 
  paste0(x, " (", round(aggregated_data_rf_subset[aggregated_data_rf_subset$dataset == x, "R2Score"], 2), ")"))

collected_aggregated_data = collected_aggregated_data[collected_aggregated_data$R2Score > 0, ]
collected_aggregated_data_ = collected_aggregated_data[collected_aggregated_data$R2Score_rel >= 0, ]


ff = ggplot(collected_aggregated_data_, aes(x = dataset, y = R2Score_rel, color = model)) + 
  geom_hline(yintercept = 0, color = "grey") + 
  geom_point(aes(shape = pasted), size = 5) +  # Larger markers
  theme_bw() + 
  xlab("") + 
  ylim(0, 0.12) +
  ylab("R² relative to RF w/o FS") + 
  scale_color_discrete(name = "ML model") +
  scale_shape_manual(values = 1:7, name = "FS approach") +
  theme(
    axis.title.y = element_text(size = 14, face = "bold"),  # Y-axis label bold and larger
    axis.text.y = element_text(size = 13, face = "bold", color = "black"), # Y-axis tick values bold and larger
    axis.title.x = element_text(size = 14, face = "bold"),  # X-axis label bold and larger
    axis.text.x = element_text(angle = 90, hjust = 1, size = 13, face = "bold", color = "black", vjust=0.5), # X-axis tick values bold and larger
    legend.title = element_text(size = 13, face = "bold"),  # Legend title bold and larger
    legend.text = element_text(size = 12, face = "bold")    # Legend text bold and larger
  )

# Save the plot
ggsave(paste0(target_folder, "r2_relative_to_nofs_.png"), 
       ff, width = 10, height = 10)


#################
##Is normal or relative better?

absolute_data = read.csv(paste0(basic_folder, "results_regression/collected.csv"), row.names = 1)
relative_data = read.csv(paste0(basic_folder, "results_regression_fraction/collected.csv"), row.names = 1)

counter = 1
for (dataset in unique(absolute_data$dataset)){
  print(dataset)
  absolute_subset = absolute_data[absolute_data$dataset == dataset,]
  absolute_RF = absolute_subset[(absolute_subset$model == "RandomForestRegressor") & (absolute_subset$selector == "None"), ]
  absolute_subset = absolute_subset[absolute_subset[, "R2Score"] == max(absolute_subset[, "R2Score"]), ]
  absolute_subset$type = "absolute"
  
  relative_subset = relative_data[relative_data$dataset == dataset,]
  relative_RF = relative_subset[(relative_subset$model == "RandomForestRegressor") & (relative_subset$selector == "None"), ]
  relative_subset = relative_subset[relative_subset[, "R2Score"] == max(relative_subset[, "R2Score"]), ]
  relative_subset$type = "relative"
  
  if (counter == 1){
    absolute_RF$type = "absolute"
    relative_RF$type = "relative"
    all_collected = rbind(absolute_subset, relative_subset)
    all_rf = rbind(absolute_RF, relative_RF)
  }else{
    all_collected = rbind(all_collected, absolute_subset)
    all_collected = rbind(all_collected, relative_subset)
    if (nrow(relative_RF) != 0){
      absolute_RF$type = "absolute"
      relative_RF$type = "relative"
      all_rf = rbind(all_rf, absolute_RF)
      all_rf = rbind(all_rf, relative_RF)
    }
  }
  counter = counter + 1
}
rm(absolute_subset, relative_subset, counter, dataset, relative_RF, absolute_RF)

##which is better: the best relative or absolute result?
best_model_comparison = ggplot(all_collected, aes(x = dataset, y = R2Score, color = model)) + 
  geom_hline(yintercept = 0, color = "lightgrey") + 
  geom_point(data = all_collected[all_collected$type == "absolute", ], color = "lightgrey", size = 4) +  # Marker size
  geom_point(aes(shape = selector), size = 4) +  # Marker size
  xlab("") +
  ylab("R² of Best Models") +  # Add Y-axis label
  scale_shape(name = "FS Method") +
  scale_color_discrete(name = "ML Model") +
  theme_bw() + 
  theme(
    axis.title.y = element_text(size = 14, face = "bold"),       # Bold and larger Y-axis label
    axis.text.y = element_text(size = 13, face = "bold", color = "black"),  # Bold and larger Y-axis tick values
    axis.text.x = element_text(angle = 90, hjust = 1, vjust=0.5, size = 13, face = "bold", color = "black"),  # Bold X-axis tick values
    legend.title = element_text(size = 13, face = "bold"),       # Bold and larger legend title
    legend.text = element_text(size = 12, face = "bold")         # Bold and larger legend text
  )



###also only plot the RF~None and compare the two 

best_rfs_comparison = ggplot(all_rf, aes(x = dataset, y = R2Score)) + 
  geom_hline(yintercept = 0, color = "lightgrey") + 
  geom_point(data = all_rf[all_rf$type == "absolute", ], color = "lightgrey", size = 4) +  # Marker size
  geom_point(shape = 15, color = "#00BFC4", size = 4) +  # Marker size
  ylim(-55, 1) + 
  ylab("R² of RF Models w/o FS") +  # Add Y-axis label
  xlab("") +
  theme_bw() + 
  theme(
    axis.title.y = element_text(size = 14, face = "bold"),       # Bold and larger Y-axis label
    axis.text.y = element_text(size = 13, face = "bold", color = "black"),  # Bold and larger Y-axis tick values
    axis.text.x = element_text(angle = 90, hjust = 1, vjust =0.5, size = 13, face = "bold", color = "black"),  # Bold X-axis tick values
    legend.title = element_text(size = 13, face = "bold"),       # Bold and larger legend title
    legend.text = element_text(size = 12, face = "bold"),        # Bold and larger legend text
    legend.position = "none"                                    # Remove legend
  )



ggsave(paste0(target_folder, "comparison_absolute_relative.png"), 
       best_model_comparison + best_rfs_comparison + plot_annotation(tag_levels = 'A'),
       width = 10, height = 10)


###########
##plot the runtimes without ranking

reg_result_data = read.csv("/home/sperlea/Desktop/project_benchmark/data/final/results_regression/collected.csv", row.names = 1)

runtime_regression = ggplot(reg_result_data, aes(total, selector)) + geom_point() + facet_grid(model~dataset, scales="free") + theme_bw() +
  ylab("FS method") + xlab("total runtime (s)") + theme(axis.text.x = element_text(angle = 45, hjust=1)) + ggtitle("Regression")

cla_result_data = read.csv("/home/sperlea/Desktop/project_benchmark/data/final/results_regression_fraction/collected.csv", row.names = 1)

runtime_regression_rel = ggplot(cla_result_data, aes(total, selector)) + geom_point() + facet_grid(model~dataset, scales="free") + theme_bw() +
  ylab("FS method") + xlab("total runtime (s)") + theme(axis.text.x = element_text(angle = 45, hjust=1)) + ggtitle("Regression, relative data")

rer_result_data = read.csv("/home/sperlea/Desktop/project_benchmark/data/final/results_classif/collected.csv", row.names = 1)

runtime_classification = ggplot(rer_result_data, aes(total, selector)) + geom_point() + facet_grid(model~dataset, scales="free") + theme_bw() +
  ylab("FS method") + xlab("total runtime (s)") + theme(axis.text.x = element_text(angle = 45, hjust=1)) + ggtitle("Classification")

ggsave(paste0(target_folder, "comparison_runtime_absolute.png"), 
       runtime_regression / runtime_regression_rel / runtime_classification + plot_annotation(tag_levels = 'A'),
       width = 20, height = 15)


total_runtimes = c(reg_result_data$total, cla_result_data$total, rer_result_data$total)
sum(total_runtimes<=60) / length(total_runtimes)

#############
## plot the boso_fish results

library(plyr)
library(ggplot2)
library(patchwork)

base_folder = "/home/sperlea/Desktop/project_benchmark/data/configs_25_02_21_boso_fish/results"

read_and_prepare_dataset = function(input_path, type_string, legend = TRUE){
  results = read.csv(input_path, row.names = 1)
  results$type = type_string
  results$selector = gsub("GenericUnivariateSelect", "GUS", results$selector)
  results$FS = paste(results$model, results$selector,
                            sapply(strsplit(results$params, "score_func::"), function(x) {ifelse(length(x) == 2, x[2], "")} ))
  results$FS = factor(results$FS,
                             levels = results$FS[order(results$ranked)])
  
  fig_class = ggplot(results, aes(FS, ranked, color = model)) + 
    geom_point() + ggtitle(type_string) + ylab("rank") + xlab("") +
    theme_bw()
  
  
  if (legend) {
    fig_class = fig_class + 
      scale_color_discrete(name = "ML model") +
      theme(
        axis.title.x = element_blank(),                        # Remove X-axis title
        axis.title.y = element_text(size = 12, face = "bold"), # Bold and larger Y-axis label
        axis.text.y = element_text(size = 10, face = "bold", color = "black"), # Bold and larger Y-axis tick values
        axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5,      # Bold and larger X-axis text
                                   size = 14, face = "bold", color = "black"),
        legend.title = element_text(size = 14, face = "bold"), # Bold and larger legend title
        legend.text = element_text(size = 14, face = "bold"),  # Bold and larger legend text
        panel.grid.major = element_blank(), 
        panel.grid.minor = element_blank(),
        axis.line.y = element_line(color = "black")
      )
  } else {
    fig_class = fig_class + 
      theme(
        axis.title.x = element_blank(),                        # Remove X-axis title
        axis.title.y = element_text(size = 12, face = "bold"), # Bold and larger Y-axis label
        axis.text.y = element_text(size = 10, face = "bold", color = "black"), # Bold and larger Y-axis tick values
        axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5,      # Bold and larger X-axis text
                                   size = 14, face = "bold", color = "black"),
        legend.position = "none",                              # Remove legend
        panel.grid.major = element_blank(), 
        panel.grid.minor = element_blank(),
        axis.line.y = element_line(color = "black")
      )
  }
  
  return(fig_class)
}

class_absolute_results = read_and_prepare_dataset(paste0(base_folder, "/cfg_class_absolute/all_ranks.csv"), "classification, absolute",
                                                  FALSE)
class_relative_results = read_and_prepare_dataset(paste0(base_folder, "/cfg_class_fraction/all_ranks.csv"), "classification, relative",
                                                  TRUE)

salinity_absolute_results = read_and_prepare_dataset(paste0(base_folder, "/cfg_reg_absolute_salinity/all_ranks.csv"), "regression (salinity), absolute",
                                                     FALSE)
watertemp_absolute_results = read_and_prepare_dataset(paste0(base_folder, "/cfg_reg_absolute_watertemp/all_ranks.csv"), "regression (temperature), absolute",
                                                      FALSE)
salinity_relative_results = read_and_prepare_dataset(paste0(base_folder, "/cfg_reg_fraction_salinity/all_ranks.csv"), "regression (salinity), relative",
                                                     TRUE)
watertemp_relative_results = read_and_prepare_dataset(paste0(base_folder, "/cfg_reg_fraction_watertemp/all_ranks.csv"), "regression (temperature), relative",
                                                      TRUE)


combined_fig = (watertemp_absolute_results | watertemp_relative_results) / 
  (salinity_absolute_results | salinity_relative_results) / 
  ( class_absolute_results  | class_relative_results ) + plot_annotation(tag_levels = 'A')


ggsave("/home/sperlea/Desktop/project_benchmark/figures/final_plots/boso_fish.png", combined_fig,  width = 18,
       height = 21)

