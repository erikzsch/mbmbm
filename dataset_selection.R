library(data.table)
library(ggplot2)
library(reshape2)
library(vegan)
library(readxl)
library(dplyr)

setwd("...")

boxplot_for_outliers = function(input_df, target_selection_folder, dataset_name){
  write.table(summary(merged_metadata), 
              paste(target_selection_folder, dataset_name, "summary.csv", sep = "/"))
  
  input_df = input_df[unlist(lapply(input_df, is.numeric))]
  melted_input_df = melt(as.matrix(input_df))
  
  boxplot_plot = ggplot(melted_input_df) + aes(x = Var2, y = value) + geom_boxplot() + theme_minimal() +
    theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))
  
  ggsave(paste(target_selection_folder, dataset_name, "boxplot.png", sep = "/"), boxplot_plot,
         width = 8, height = 6, bg = "white")
  
  all_outlier_ids = c()
  for (cc in colnames(input_df)){
    out = boxplot.stats(input_df[[cc]])$out
    out_ind <- which(input_df[[cc]] %in% c(out))
    all_outlier_ids = c(out_ind)
  }
  counted_outlier_ids = table(all_outlier_ids)
  
  names(counted_outlier_ids) = rownames(merged_metadata)[as.numeric(names(counted_outlier_ids))]
  write.table(counted_outlier_ids, paste(selection_target_folder, dataset_name, "outlier_counted.csv", sep = "/"))
}

rarefaction_curve = function(otu_data, selection_target_folder, dataset_name, filename = "rarefaction.png"){
  png(paste(selection_target_folder, dataset_name, filename, sep = "/"), width = 1000, height = 1000) 
  rarecurve(otu_data, step = 300, sample = min(rowSums(otu_data)), col = "blue", cex = 0.6)
  dev.off() 
}

plot_bc_nmds_plots = function(otu_table, metadata, start_int_rownames, selection_target_folder, dataset_name,
                              file_name_ending = ".png"){
  set.seed(123)
  nmds = metaMDS(otu_table, distance = "bray")
  data.scores = as.data.frame(scores(nmds)$sites)
  data.scores = merge(data.scores, metadata[row.names(data.scores), ], by = 0)
  for (cc in colnames(data.scores)[start_int_rownames:length(colnames(data.scores))]){
    xx = ggplot(data.scores, aes_string(x = "NMDS1", y = "NMDS2", color = cc)) + geom_point(size = 4) 
    ggsave(paste(selection_target_folder, "/", dataset_name, "/", "bc_nmds_", cc, file_name_ending, sep = ""),
           width = 6, height = 6)
  }
}

multiplesheets <- function(fname) {
  
  # getting info about all excel sheets
  sheets <- readxl::excel_sheets(fname)
  tibble <- lapply(sheets, function(x) readxl::read_excel(fname, sheet = x))
  data_frame <- lapply(tibble, as.data.frame)
  
  # assigning names to data frames
  names(data_frame) <- sheets
  
  # print data frame
  return(data_frame)
}

dataset_target_folder = "./dataset_selection"
selection_target_folder = "./dataset_selection"
clean_target_folder = "./clean"

###############
######tara oceans
## merge the metadata tables that were created from subtable W1 and W8 from the original metadata (OM.CompanionTables.xlsx)
comp_w1 = read.csv("./raw/tara_oceans/companion_table_W1.csv")
comp_w8 = read.csv("./raw/tara_oceans/companion_table_W8.csv")

##which columns to keep
comp_w1$size.fraction = paste(comp_w1$Size.fraction.lower.threshold..micrometre., comp_w1$Size.fraction.upper.threshold..micrometre., sep = "--")
rownames(comp_w1) = comp_w1$PANGAEA.sample.identifier
colnames_w1 = c("Station.identifier..TARA_station..", "Sample.label..TARA_station._environmental.feature_size.fraction.", 
                "Environmental.Feature", "size.fraction", "Marine.pelagic.biomes..Longhurst.2007.")
comp_w1 = comp_w1[colnames_w1]
###all the PANGAEA.sample.ids from comp_w1 are in comp_w8
##comp_w1$PANGAEA.sample.identifier %in% comp_w8$PANGAEA.Sample.ID
rownames(comp_w8) = comp_w8$PANGAEA.Sample.ID
comp_w8 = comp_w8[2:13]
merged_metadata = merge(comp_w1, comp_w8, by = 0)

colnames(merged_metadata) = sapply(colnames(merged_metadata), function(x) strsplit(x, "..", fixed=T)[[1]][1])
merged_metadata$Sample.label = make.names(merged_metadata$Sample.label)
rm(comp_w1, comp_w8, colnames_w1)

#####read otu table
otu_data = read.csv("./raw/tara_oceans/miTAG.taxonomic.profiles.release.tsv", 
                    sep = "\t")
taxonomy = otu_data[1:7]
otu_data = otu_data[8:length(colnames(otu_data))]
rownames(taxonomy) = paste("OTU_", rownames(taxonomy), sep = "")
rownames(otu_data) = paste("OTU_", rownames(otu_data), sep = "")

otu_data = as.data.frame(t(otu_data))

##subset metadata to contain only samples with otus
merged_metadata = merged_metadata[merged_metadata$Sample.label %in% rownames(otu_data), ]


####** note that some values are equal or below the detection limit of 0,02umol/L and should be interpretated as <=0,02umol/L
###remove everything below Limit of Detection (or quantification or whatever) for 
for (i in c(14:17)){
  merged_metadata[[i]][merged_metadata[i] <= 0.02] <- NA
}
rm(i)
row.names(merged_metadata)= merged_metadata$Sample.label

##count NAs, outliers and plot outliers
boxplot_for_outliers(merged_metadata, selection_target_folder, "tara_oceans")

## write metadata to file
write.csv(merged_metadata, paste(clean_target_folder, "tara_oceans/metadata.csv", sep = "/"))

## write taxonomy to file
write.csv(taxonomy, paste(clean_target_folder, "tara_oceans/taxonomy.csv", sep = "/"))

##write otu table to file
write.csv(otu_data, paste(clean_target_folder, "tara_oceans/otus.csv", sep = "/"))

###look at Bray-Curtis
plot_bc_nmds_plots(otu_data, merged_metadata, 8, selection_target_folder, "tara_oceans")

##rarefaction curve
rarefaction_curve(otu_data, selection_target_folder, "tara_oceans")

rm(merged_metadata, otu_data, taxonomy)

##end of this dataset

###############
######nz_springs

metadata_table = read.csv("./raw/nz_springs/20190307_925springs_physicochemistry.csv")
otu_table = read.csv("./raw/nz_springs/R1-R23_OTUtableCurated-1.csv")

row.names(metadata_table) = metadata_table[[1]]
metadata_table = metadata_table[-1]

otu_table = otu_table[1:(length(row.names(otu_table))-4), ]
row.names(otu_table) = otu_table[[1]]
taxonomy = otu_table[969:983]
otu_table = as.data.frame(t(otu_table[2:967]))
row.names(otu_table) = sapply(row.names(otu_table), function(x) strsplit(x, "_")[[1]][1])  

common_sample_ids = row.names(metadata_table)[row.names(metadata_table) %in% row.names(otu_table)]

otu_table = otu_table[common_sample_ids, ]
metadata_table = metadata_table[common_sample_ids, ]
rm(common_sample_ids)

###remove metadata columns that are not of interest
colnames(metadata_table)
remove_columns = c("size", "colour", "ebullition", "dnaVolume", "soilCollected")
metadata_table = metadata_table[!(colnames(metadata_table) %in% remove_columns)]

#remove metadata with "<", then remove samples with too little 
for (cc in colnames(metadata_table)){
  metadata_table[[cc]][grepl("<", metadata_table[[cc]])] = NA
  metadata_table[[cc]] = as.numeric(metadata_table[[cc]])
}
na_count = colSums(is.na(metadata_table))/length(row.names(metadata_table))
metadata_table = metadata_table[na_count<0.1] ##This is a strong limit, but looks good enough!

##rarefaction curve
rarefaction_curve(otu_table, selection_target_folder, "nz_springs")

### there are massive differences in the rarefaction....
###P1.0443 is an outlier wrt. total abundance and I'll remove it
remove_index = which(rowSums(otu_table) == max(rowSums(otu_table)))
otu_table = otu_table[-remove_index, ]
metadata_table = metadata_table[-remove_index,]

rarefaction_curve(otu_table, selection_target_folder, "nz_springs")

###look at Bray-Curtis
plot_bc_nmds_plots(otu_table, metadata_table, 4, selection_target_folder, "nz_springs")

## write metadata to file
write.csv(metadata_table, paste(clean_target_folder, "nz_springs/metadata.csv", sep = "/"))

## write taxonomy to file
write.csv(taxonomy, paste(clean_target_folder, "nz_springs/taxonomy.csv", sep = "/"))

##write otu table to file
write.csv(otu_table, paste(clean_target_folder, "nz_springs/otus.csv", sep = "/"))

###############
##eu_lakes

otu_table_prok = data.frame(fread("./raw/eu_lakes/bact_otus.csv"), row.names = 1)
otu_table_euk = data.frame(fread("./raw/eu_lakes/euk_otus.csv"), row.names = 1)
metadata = read.csv("./raw/eu_lakes/physchem_all.csv", row.names = 1)

removecols = c("Date", "Country", "Time")
metadata = metadata[!(colnames(metadata) %in% removecols)]

shared_samples_prok = rownames(otu_table_prok)[rownames(otu_table_prok) %in% rownames(metadata)]
shared_samples_euk = rownames(otu_table_euk)[rownames(otu_table_euk) %in% rownames(metadata)]
shared_samples = shared_samples_prok[shared_samples_prok %in% shared_samples_euk]

metadata = metadata[shared_samples, ]
otu_table_prok = otu_table_prok[shared_samples, ]
colnames(otu_table_prok) = paste(colnames(otu_table_prok), "_prok", sep = "")
otu_table_euk = otu_table_euk[shared_samples, ]
colnames(otu_table_prok) = paste(colnames(otu_table_prok), "_euk", sep = "")

otu_table = merge(otu_table_prok, otu_table_euk, by=0)
rownames(otu_table) = otu_table[, 1]
otu_table = otu_table[, -1]
otu_abundance = colSums(otu_table) 
otu_table = otu_table[otu_abundance != 0] ###remove all all-zero OTUs

rm(otu_abundance, otu_table_prok, otu_table_euk, removecols, shared_samples_euk, shared_samples_prok, shared_samples)

metadata_rownas = rowSums(is.na(metadata))
metadata_colnas = colSums(is.na(metadata))

metadata_allfeatures = metadata[metadata_rownas == 0, ]
metadata_allsamples =  metadata[, metadata_colnas == 0]
otu_table_allfeatures = otu_table[rownames(metadata_allfeatures),]
otu_table_allsamples = otu_table[rownames(metadata_allsamples),]

###
rm(otu_table, metadata, metadata_colnas, metadata_rownas)

rarefaction_curve(otu_table_allfeatures, selection_target_folder, "eu_lakes", "rarefaction_allfeatures.png")
rarefaction_curve(otu_table_allsamples, selection_target_folder, "eu_lakes", "rarefaction_allsamples.png")

plot_bc_nmds_plots(otu_table_allfeatures, metadata_allfeatures, 4, selection_target_folder, "eu_lakes", "_allfeatures.png")
plot_bc_nmds_plots(otu_table_allsamples, metadata_allsamples, 4, selection_target_folder, "eu_lakes", "_allsamples.png")

##the rarefaction curves look bad.

## write metadata to file
write.csv(metadata_allfeatures, paste(clean_target_folder, "eu_lakes_allfeatures/metadata.csv", sep = "/"))
write.csv(metadata_allsamples, paste(clean_target_folder, "eu_lakes_allsamples/metadata.csv", sep = "/"))

##write otu table to file
write.csv(otu_table_allfeatures, paste(clean_target_folder, "eu_lakes_allfeatures/otus.csv", sep = "/"))
write.csv(otu_table_allsamples, paste(clean_target_folder, "eu_lakes_allsamples/otus.csv", sep = "/"))

## write taxonomy to file
###go through taxonomy line by line and create subsets, do the correct naming change
fileName = "./raw/eu_lakes/all_taxonomy.csv"
target_filename = "./clean/eu_lakes_allsamples/taxonomy.csv"

con = file(fileName, open="r")
target_con = file(target_filename)
counter = 1
while ( TRUE ) {
  line = readLines(con, n = 1)
  
  if ( length(line) == 0 ) {
    break
  }

  line = gsub("\"", "", line, fixed = T)
  line = strsplit(line, ",")[[1]]
  
  if (counter == 1){
    new_line = line
  }else if (line[2] == "Bacteria"){
    this_id = paste(line[1], "_prok", sep = "")
    new_line = paste(c(this_id, line[2:length(line)]), collapse = ", ")
  }else if (line[2] == "Archaea"){
    this_id = paste(line[1], "_prok", sep = "")
    new_line = paste(c(this_id, line[2:length(line)]), collapse = ", ")
  }else if (line[2] == "Eukaryota"){
    this_id = paste(line[1], "_euk", sep = "")
    new_line = paste(c(this_id, line[2:length(line)]), collapse = ", ")
  }else{
    NA
  }
  cat(new_line, file = target_filename, sep = "\n", append = T)
  counter = counter + 1
}
close(con)
close(target_con)


###############
##atl_ocean_transect
otu_xls_file = multiplesheets("./raw/atl_ocean_transect/41598_2016_BFsrep19054_MOESM2_ESM.xls")
metadata_table = as.data.frame(read_excel("./raw/atl_ocean_transect/41598_2016_BFsrep19054_MOESM3_ESM.xls"))

collected_sequences = c()
counter = 1
for (sheetname in names(otu_xls_file)[2:4]){
  print(sheetname)
  
  short_name = strsplit(sheetname, " ")[[1]][3]
  this_df = otu_xls_file[[sheetname]]
  this_df$OTU = paste(short_name, this_df$OTU, sep = "_")
  rownames(this_df) = this_df$OTU
  
  these_sequences = this_df[, "Sequence"]
  names(these_sequences) = rownames(this_df) 
  this_df = this_df[, 2:(length(this_df[1, ])-2)]
  this_df = as.data.frame(t(this_df))
  
  collected_sequences = c(collected_sequences, these_sequences)
  
  if (counter == 1){
    collected_otus = this_df
  }else{
    collected_otus = merge(collected_otus, this_df, by = 0)
    rownames(collected_otus) = collected_otus$Row.names
    collected_otus = collected_otus[, -1]
  }
  counter = counter + 1
}

collected_sequences = as.data.frame(collected_sequences)
## write sequences to file
write.csv(collected_sequences, paste(clean_target_folder, "atl_ocean_transect/sequences.csv", sep = "/"))
rm(collected_sequences, otu_xls_file, this_df, counter, short_name, sheetname, these_sequences)

rownames(metadata_table) = metadata_table$`Sample ID`
metadata_table = metadata_table[2:7]
common_sample_ids = rownames(metadata_table)[rownames(metadata_table) %in% rownames(collected_otus)]

collected_otus = collected_otus[common_sample_ids, ]
metadata_table = metadata_table[common_sample_ids, ]

colnames(metadata_table) = sapply(colnames(metadata_table), function(x) strsplit(x, " ")[[1]][1])

rarefaction_curve(collected_otus, selection_target_folder, "atl_ocean_transect")

###look at Bray-Curtis
plot_bc_nmds_plots(collected_otus, metadata_table, 4, selection_target_folder, "atl_ocean_transect")

## write metadata to file
write.csv(metadata_table, paste(clean_target_folder, "atl_ocean_transect/metadata.csv", sep = "/"))

##write otu table to file
write.csv(collected_otus, paste(clean_target_folder, "atl_ocean_transect/otus.csv", sep = "/"))

###############
##bog_lakes

library(OTUtable)

data(metadata)
data(otu_table)

otu_table = as.data.frame(t(otu_table))
otu_table = otu_table[!grepl(".R2", rownames(otu_table), fixed = T), ]
otu_table = otu_table[!grepl(".R3", rownames(otu_table), fixed = T), ]
otu_table = otu_table[!grepl(".1", rownames(otu_table), fixed = T), ]

##this is all from https://github.com/McMahonLab/North_Temperate_Lakes-Microbial_Observatory/blob/master/Scripts%2BWorkflows/Analysis_scripts/manuscript_plots_accepted_2017-06-07.R
metadata <- metadata[,c(1,2,3,4,5,6)]
metadata$Layer <- substr(metadata$Sample_Name, start = 3, stop = 3)
metadata <- metadata[which(metadata$Layer == "E"), ]
metadata$Site <- substr(metadata$Sample_Name, start = 1, stop = 2)
metadata$Date <- extract_date(metadata$Sample_Name)
layer <- c()
for(i in 1:dim(metadata)[1]){
  sample <- metadata[i, ]
  if(sample$Site == "CB" | sample$Site == "FB"){
    if(sample$Depth <= 1){
      layer[i] <- "Epi"
    }else if(sample$Depth > 1){
      layer[i] <- "Hypo"
    }
  }else if (sample$Site == "WS" | sample$Site == "NS" | sample$Site == "SS" | sample$Site == "TB"| sample$Site == "HK" | sample$Site == "MA"){
    if(sample$Depth <= 2){
      layer[i] <- "Epi"
    }else if(sample$Depth > 2){
      layer[i] <- "Hypo"
    }
  }
}
metadata$Layer <- layer
metadata$Depth <- NULL
mean_meta2 <- aggregate(x = metadata, by = list(metadata$Layer, metadata$Site, metadata$Sample_Name), FUN = "mean")
mean_meta2 <- mean_meta2[which(is.na(mean_meta2$Temperature) == F), ]

rm(metadata, sample, i, layer)

prefix =  substr(mean_meta2$Group.3, start = 1, stop = 2)
middle_part = prefix
middle_part[mean_meta2$Group.1 == "Epi"] = "E"
middle_part[mean_meta2$Group.1 == "Hypo"] = "H"
suffix =  substr(mean_meta2$Group.3, start = 4, stop = 10)
mean_meta2$Group.3 = sapply(c(1:length(prefix)), function(x) paste(prefix[x], middle_part[x], suffix[x], sep = ""))
rownames(mean_meta2) = mean_meta2$Group.3

mean_meta2$year = substr(rownames(mean_meta2), start=9, stop=10)
mean_meta2$month = substr(rownames(mean_meta2), start=6, stop=8)

mean_meta2 = mean_meta2[!(colnames(mean_meta2) %in% c("Group.3", "Sample_Name", "Layer", "Site"))]
rm(prefix, suffix, middle_part)

colnas = colSums(is.na(mean_meta2))/length(rownames(mean_meta2))
mean_meta2 = mean_meta2[colnas<0.5]

colnames(mean_meta2)[1] = "Depth"
colnames(mean_meta2)[2] = "Lake_ID"

rownames(otu_table) = gsub(".R1", "", rownames(otu_table))

ids_in_both = rownames(otu_table)[rownames(otu_table) %in% rownames(mean_meta2)]

otu_table = otu_table[ids_in_both, ]
meta_table = mean_meta2[ids_in_both, ]
rm(mean_meta2, colnas, ids_in_both, otu_sample_ids, rownas)

rarefaction_curve(otu_table, selection_target_folder, "bog_lakes")

###look at Bray-Curtis
plot_bc_nmds_plots(otu_table, meta_table, 4, selection_target_folder, "bog_lakes")

## write metadata to file
write.csv(meta_table, paste(clean_target_folder, "bog_lakes/metadata.csv", sep = "/"))

##write otu table to file
write.csv(otu_table, paste(clean_target_folder, "bog_lakes/otus.csv", sep = "/"))


###############
##wastewater_treatment

otu_table = as.data.frame(fread("./raw/wastewater_treatment/GWMC_16S_otutab.txt"))
metadata_table = as.data.frame(read_excel("./raw/wastewater_treatment/41564_2019_426_MOESM3_ESM.xlsx"))

row.names(otu_table) = otu_table[[1]]
otu_table = otu_table[-1]
otu_table = data.frame(t(otu_table))

metadata_table = metadata_table[!(colnames(metadata_table) %in% c("...2", "...3", "...4", "...5", "...6", "...7", "...8", "...9", "...16", 
  "...25", "...26", "...28", "...29", "...30", "...31", "...32", "...33", 
  "...33", "...34", "...35", "...36", "...37", "...38", "...39", "...40", 
  "...41", "...45", "...46", "...47", "...51", "...52")) ]
##create column names
new_colnames = c()
for (i in c(1:length(metadata_table))){
  pot_name = metadata_table[4, i]
  if (is.na(pot_name)){
    pot_name = metadata_table[3, i]
    if (is.na(pot_name)){
      pot_name = metadata_table[2, i]
    }
  }
  new_colnames = c(new_colnames, pot_name)
}

new_colnames[4] = "Temperature Annual average"
new_colnames[5] = "Temperature Annual mean of daily maximum"
new_colnames[6] = "Temperature Annual mean of daily minimum"
new_colnames[7] = "Temperature Sampling month average"
new_colnames[8] = "Precipitation (mm) Annual"
new_colnames[9] = "Precipitation (mm) Sampling month"
new_colnames[17] = "NH4-N (mg/l) Inf"
new_colnames[18] = "NH4-N (mg/l) Aeration tank inf"
new_colnames[19] = "NH4-N (mg/l) Eff"
new_colnames[20] = "TP (mg/l)) Inf"
new_colnames[21] = "TP (mg/l) Aeration tank inf"
new_colnames[22] = "TP (mg/l) Eff"
colnames(metadata_table) = new_colnames

###delete the first 5 rows or so, and the last 4 as they're empty
metadata_table = metadata_table[-(1:5),]
metadata_table = metadata_table[-(1187:1190),]

##create a subset that contains not too many NAs
metadata_table[ metadata_table == "NA" ] <- NA
colnas = colSums(is.na(metadata_table))/length(rownames(metadata_table))
metadata_table = metadata_table[colnas < 0.4]

subset_metadata = metadata_table[20:22]
metadata_table = metadata_table[rowSums(is.na(subset_metadata)) == 0, ]
colnas = colSums(is.na(metadata_table))/length(rownames(metadata_table))
metadata_table = metadata_table[colnas == 0]

rownames(metadata_table) = metadata_table[[1]]
metadata_table = metadata_table[-1]
rm(colnas, i, new_colnames, pot_name, subset_metadata)

common_sample_ids = rownames(otu_table)[rownames(otu_table) %in% rownames(metadata_table)]
##remove samples that are outliers based on NMDS
remove_ids = c("USNO2B", "USNO2C", "USNO2D", "USNO2E", "USNO2F", "USOK01A", 
               "USOK01B", "USOK01C", "USOK02A", "USOK02B", "USOK02C", "USOK03A", 
               "USOK03B", "USOK03C", "USOK03D", "COCL2A", "COCL2B")
common_sample_ids = common_sample_ids[!(common_sample_ids %in% remove_ids)]
otu_table = otu_table[common_sample_ids, ]
otu_table = otu_table[colSums(otu_table) != 0]
metadata_table = metadata_table[common_sample_ids, ]
rm(common_sample_ids)

colnames(metadata_table) = make.names(colnames(metadata_table))

rarefaction_curve(otu_table, selection_target_folder, "wastewater_treatment")

##make the columns in metadata_table numeric!
for (i in 3:10){
  metadata_table[i] = as.numeric(metadata_table[[i]])
}

###look at Bray-Curtis
plot_bc_nmds_plots(otu_table, metadata_table, 4, selection_target_folder, "wastewater_treatment")

## write metadata to file
write.csv(metadata_table, paste(clean_target_folder, "wastewater_treatment/metadata.csv", sep = "/"))

##write otu table to file
write.csv(otu_table, paste(clean_target_folder, "wastewater_treatment/otus.csv", sep = "/"))

###############
##australia


otu_table = data.frame(fread("./raw/australia/d__Bacteria.csv"))
full_taxonomy = apply(otu_table[c(2, 5, 6, 7, 8, 9, 10, 11)], 1, function(x) paste(x, collapse = "\t"))
unique_taxonomy = as.data.frame(t(as.data.frame(strsplit(unique(full_taxonomy), "\t"))))
rownames(unique_taxonomy) = paste("otu", 1:length(unique_taxonomy[, 1]), sep = "_")
rm(full_taxonomy)

otu_table = otu_table[c(1, 2, 3)]
for (otu_id in rownames(unique_taxonomy)){
  print(otu_id)
  otu_table$OTU[otu_table$OTU == unique_taxonomy[otu_id, 1]] = otu_id
}

## write taxonomy to file
write.csv(unique_taxonomy, paste(clean_target_folder, "australia/taxonomy.csv", sep = "/"))

## write temporary otu table to file
otu_table = otu_table[-c(4, 5, 6, 7, 8, 9, 10, 11, 12)]
write.csv(otu_table, 
          "./raw/australia/d__Bacteria_semiraw_long.csv")
rm(unique_taxonomy, otu_id)

otu_table_wide = dcast(otu_table, Sample.ID ~ OTU)
otu_table_wide[is.na(otu_table_wide)] <- 0
otu_table = otu_table_wide
rm(otu_table_wide)
##write otu table to file
write.csv(otu_table, paste(clean_target_folder, "australia/otus_initial.csv", sep = "/"))
rm(otu_table)
#unique_otu_sequences = unique(otu_table$OTU)

##look at the metadata now while the otu_table is not loaded
metadata = read.csv("./raw/australia/contextual.csv")

##the authors of the data have thought it a good idea to use both NAs and -9999
metadata[metadata == -9999] = NA
colnas = colSums(is.na(metadata))/length(rownames(metadata))
metadata = metadata[names(colnas)[colnas < 0.8]]

keep_colnames = c("Sample.ID", "Alkalinity..µmol.kg.", "Ammonium..µmol.L.", "Chlorophyll.A..mg.m3.", "Chlorophyll.Ctd..mg.m3.", "Conductivity.Aqueous..S.m.",
                  "Density..kg.m3.", "Depth..m.", "Nitrate..µmol.L.", "Nitrate.Nitrite..µmol.L.", "Oxygen..µmol.L.", "Phosphate..µmol.L.",
                  "Salinity..PSU.", "Secchi.Depth..m.", "Silicate..µmol.L.", "Temp..degC.", "Tot.Depth.Water.Col..m.", "Total.Co2..µmol.kg.",
                  "Turbidity..NTU_or_FTU." )
location_colnames = c("Sample.ID", "Collection.Date", "Geo.Loc.Name", "General.Env.Feature", "Sample.Type")
metadata_loc = metadata[location_colnames]
metadata = metadata[keep_colnames]
rm(keep_colnames, location_colnames, colnas)

##remove some locations completely
remove_locations = c("Southern Ocean: K-axis", "Southern Ocean: Totten", "Australia:Perth", "Australia: Queensland: Orpheus Island:Channel",
                     "Australia:New South Wales", "Australia:Western Australia", "Australia: Queensland: Magnetic Island:Geoffrey Bay",
                     "Australia: Queensland: Orpheus Island: Pioneer Bay")
metadata_loc = metadata_loc[!(metadata_loc$Geo.Loc.Name %in% remove_locations), ]
metadata = metadata[metadata$Sample.ID %in% metadata_loc$Sample.ID, ]
rm(remove_locations)

colnas = colSums(is.na(metadata))/length(rownames(metadata))

subset_metadata = metadata[colnas < 0.3]
rownas = rowSums(is.na(subset_metadata))/length(colnames(subset_metadata))
subset_metadata = subset_metadata[rownas == 0, ]
subset_metadata_loc = metadata_loc[metadata_loc$Sample.ID %in% subset_metadata$Sample.ID, ]
rownames(subset_metadata) = subset_metadata[[1]]
subset_metadata = subset_metadata[-1]
rm(metadata, metadata_loc, colnas, rownas)

## write metadata to file
write.csv(subset_metadata, paste(clean_target_folder, "australia/metadata.csv", sep = "/"))
write.csv(subset_metadata_loc, paste(clean_target_folder, "australia/metadata_location.csv", sep = "/"))

##go though the OTU table again
otu_table = data.frame(fread(paste(clean_target_folder, "australia/otus_initial.csv", sep = "/")))
otu_table = otu_table[-1]
rownames(otu_table) = otu_table[[1]]
otu_table = otu_table[-1]
otu_table = otu_table[rownames(subset_metadata), ]

##rarefaction curve
rarefaction_curve(otu_table, selection_target_folder, "australia")
###look at Bray-Curtis
plot_bc_nmds_plots(otu_table, subset_metadata, 4, selection_target_folder, "australia")
plot_bc_nmds_plots(otu_table, subset_metadata_loc, 4, selection_target_folder, "australia")

##write otu table to file
write.csv(otu_table, paste(clean_target_folder, "australia/otus.csv", sep = "/"))
fwrite(otu_table, paste(clean_target_folder, "australia/otus_fwrite.csv", sep = "/"))

###############
##ports

otu_data = as.data.frame(readRDS("./raw/ports/seqtabmergedNoC.rds"))
tax_data = as.data.frame(readRDS("./raw/ports/taxa.rds"))

###create otu_ids and save sequences as df
otu_ids = c()
seqs = c()
for (i in c(1:length(otu_data))){
  print(paste(i, length(otu_data)))
  otu_ids = c(otu_ids, paste("otu_", as.character(i), sep = ""))
  seqs = c(seqs, colnames(otu_data)[i])
}
seq_data = data.frame(id = otu_ids, seq = seqs)
colnames(otu_data) = otu_ids
rownames(tax_data) = otu_ids

write.csv(seq_data, paste(clean_target_folder, "ports/sequences.csv", sep = "/"))
write.csv(tax_data, paste(clean_target_folder, "ports/taxonomy.csv", sep = "/"))

##look into metadata
meta_data = read.csv("/media/sperlea/Elements/projects/project_benchmark_microML/data/raw/ports/pm_metadata.csv")

rownames(meta_data) = meta_data$sample_name
keep_columns = c("lat", "long", "barometric_press", "conduc", "salinity", "total_dissolved_solids", "surf_temp", "ph", "redox_potential", "diss_oxygen", "depth", "sample_type")
meta_data = meta_data[colnames(meta_data) %in% keep_columns]
meta_data = meta_data[meta_data$sample_type != "blank" & meta_data$sample_type != "control" & meta_data$sample_type != "unknown", ]

#remove units, make numeric or factors
meta_data[meta_data=="not collected"] <- NA
for (i in c(1:11)){
  if (i != 8){
    print(i)
    meta_data[i] = apply(meta_data[i], 1, function(x) as.numeric(strsplit(x, " ")[[1]][1])) 
  }
}

na_rows = rowSums(is.na(meta_data))
na_cols = colSums(is.na(meta_data))
meta_data = meta_data[na_rows == 0, ]
meta_data = meta_data[!grepl("post", rownames(meta_data)), ]
retained_sample_names = rownames(meta_data)
rm(seq_data, tax_data, keep_columns, na_cols, na_rows, seqs)

####finish the otu table; separate the post from the other samples
pre_otus = otu_data[!grepl("post", rownames(otu_data)), ]
post_otus = otu_data[grepl("post", rownames(otu_data)), ]
rownames(post_otus) = sub("post", "", rownames(post_otus))

post_otus = post_otus[rownames(post_otus) %in% retained_sample_names, ]
pre_otus = pre_otus[rownames(pre_otus) %in% retained_sample_names,]

meta_post = meta_data[row.names(post_otus), ]
meta_pre = meta_data[row.names(pre_otus), ]

plot_bc_nmds_plots(post_otus, meta_post, 4, selection_target_folder, "ports/post")
plot_bc_nmds_plots(pre_otus, meta_pre, 4, selection_target_folder, "ports/pre")

#remove the sample type as it is always the same
meta_post = meta_post[colnames(meta_post) != "sample_type"]
meta_pre = meta_pre[colnames(meta_pre) != "sample_type"]

write.csv(meta_post, paste(clean_target_folder, "ports/metadata_small.csv", sep = "/"))
write.csv(meta_pre, paste(clean_target_folder, "ports/metadata_large.csv", sep = "/"))
write.csv(post_otus, paste(clean_target_folder, "ports/otus_small.csv", sep = "/"))
write.csv(pre_otus, paste(clean_target_folder, "ports/otus_large.csv", sep = "/"))

###############
##bedford_basin

##there are V4V5 and V6V7 data here and rarefied and unrarefied data. I will throw away the rarefied data, but will need to do a good amount of work on the metadata

otu_data_V4V5 = multiplesheets("./raw/bedford_basin_V4V5/43705_2022_119_MOESM4_ESM.xlsx")

date_exchange = otu_data_V4V5[["MetaData"]]
sequences_V4V5 = otu_data_V4V5[["ASV Sequences"]]
otu_data_V4V5 = otu_data_V4V5[["Dataset Unrarefied"]]
colnames(otu_data_V4V5) = otu_data_V4V5[1, ]
otu_data_V4V5 = otu_data_V4V5[-1, ]
rownames(otu_data_V4V5) = otu_data_V4V5[, 1]
otu_data_V4V5 = otu_data_V4V5[, -1]

i = 1
for (old_otu_id in row.names(otu_data_V4V5)){
  new_otu_id =  paste("otu_", as.character(i), sep = "")
  rownames(otu_data_V4V5)[i] = new_otu_id
  sequences_V4V5[which(sequences_V4V5[1] == paste(">", old_otu_id, sep = "")), 1] = paste(">", new_otu_id, sep = "")
  i = i + 1
}
rm(i, old_otu_id, new_otu_id)
otu_data_V4V5 = data.frame(t(otu_data_V4V5))
taxonomy_V4V5 = t(otu_data_V4V5[789, ])
otu_data_V4V5 = otu_data_V4V5[-789, ]

##there is a taxonomy in otu_data_v4v5
# 
#link the sample ID to the metadata
##prepare the sample id for date comparison
sample_ids = rownames(otu_data_V4V5)
year = as.numeric(sapply(strsplit(gsub("BB", "", sample_ids), "\\."), function(x) x[[1]])) + 2000
week = as.numeric(sapply(strsplit(sample_ids, "\\."), function(x) substring(x[[2]], 1, nchar(x[[2]])-1)))
depth = sapply(sample_ids, function(x) substring(x, nchar(x), nchar(x)))
numeric_depth = c()
for (d in depth){
  if (d == "A"){
    numeric_depth = c(numeric_depth, 1)
  }else if (d == "B"){
    numeric_depth = c(numeric_depth, 5)
  }else if (d == "C"){
    numeric_depth = c(numeric_depth, 10)
  }else if (d == "D"){
    numeric_depth = c(numeric_depth, 60)
  }
}
basic_metadata_V4V5 = data.frame(sample_id = sample_ids, week = week, year = year, depth = numeric_depth)

date_exchange$...3 = sapply(as.POSIXct(as.numeric(date_exchange$...3) * (60*60*24), origin="1899-12-30"), function(x) strsplit(as.character(x), " ")[[1]][1])
basic_metadata_V4V5[["full_date"]] =sapply(1:nrow(basic_metadata_V4V5), 
                                           function(x) date_exchange$...3[which(date_exchange$METADATA == basic_metadata_V4V5[x, "week"] & 
                                                date_exchange$...2 == basic_metadata_V4V5[x, "year"])])
rm(depth, numeric_depth, d, week, year, sample_ids)


##prepare the metadata table for date comparisons
metadata = read.csv("./raw/bedford_basin_V4V5/CSV/bbmp_aggregated_profiles.csv")
metadata$date = sapply(strsplit(metadata$time_string, " "), function(x) x[[1]])

#where is the depth in the metadata??
counter = 1
for (i in 1:nrow(basic_metadata_V4V5)){
  this_date = basic_metadata_V4V5$full_date[i]
  this_depth = basic_metadata_V4V5$depth[i]
  this_index = basic_metadata_V4V5$sample_id[i]
  metadata_location = which(metadata$date == this_date & metadata$pressure == this_depth)
  
  if (length(metadata_location) != 0){
    this = metadata[metadata_location, ]
    this["id"] = this_index
    
    if(counter == 1){
        completed_metadata = this
    }else{
        completed_metadata = rbind(completed_metadata, this)
    }
    counter = counter + 1
  }
}
rm(counter, this_depth, this_date, this_index, this, i, metadata_location)###rm a lot of things
rm(basic_metadata_V4V5, date_exchange, metadata)


###remove unnecessary metadata columns
rownames(completed_metadata) = completed_metadata$id
keep_columns = c("year_time", "month_time", "day_time", "pressure", "temperature", "conductivity", 
                 "fluorometer", "par", "sigmaTheta", "oxygen")
completed_metadata = completed_metadata[keep_columns]
completed_metadata = completed_metadata[rowSums(is.na(completed_metadata)) == 0, ]


##get rid of all the lines in the sequencing data I do not have metadata for
otu_data_V4V5 = otu_data_V4V5[rownames(otu_data_V4V5) %in% completed_metadata$id, ]

for (cc in colnames(otu_data_V4V5)){
  otu_data_V4V5[[cc]] = as.numeric(otu_data_V4V5[[cc]])
}
rm(cc)

##create some plots and save the stuff
plot_bc_nmds_plots(otu_data_V4V5, completed_metadata, 5, selection_target_folder, "bedford_basin/V4V5")
write.csv(completed_metadata, paste(clean_target_folder, "bedford_basin/V4V5/metadata.csv", sep = "/"))
write.csv(otu_data_V4V5, paste(clean_target_folder, "bedford_basin/V4V5/otus.csv", sep = "/"))
write.csv(taxonomy_V4V5, paste(clean_target_folder, "bedford_basin/V4V5/taxonomy.csv", sep = "/"))
write.csv(sequences_V4V5, paste(clean_target_folder, "bedford_basin/V4V5/sequences.csv", sep = "/"),
          row.names = F, quote=FALSE)

### do it again for the V6V8 thing
otu_data_V6V8 = multiplesheets("/media/sperlea/Elements/projects/project_benchmark_microML/data/raw/bedford_basin_V6V8/43705_2022_119_MOESM5_ESM.xlsx")

date_exchange = otu_data_V6V8[["MetaData"]]
sequences_V6V8 = otu_data_V6V8[["ASV Sequences"]]
otu_data_V6V8 = otu_data_V6V8[["Dataset Not Rarefied"]]
colnames(otu_data_V6V8) = otu_data_V6V8[1, ]
otu_data_V6V8 = otu_data_V6V8[-1, ]
rownames(otu_data_V6V8) = otu_data_V6V8[, 1]
otu_data_V6V8 = otu_data_V6V8[, -1]

i = 1
for (old_otu_id in row.names(otu_data_V6V8)){
  new_otu_id =  paste("otu_", as.character(i), sep = "")
  rownames(otu_data_V6V8)[i] = new_otu_id
  sequences_V6V8[which(sequences_V6V8[1] == paste(">", old_otu_id, sep = "")), 1] = paste(">", new_otu_id, sep = "")
  i = i + 1
}
rm(i, old_otu_id, new_otu_id)
otu_data_V6V8 = data.frame(t(otu_data_V6V8))
taxonomy_V6V8 = t(otu_data_V6V8[777, ])
otu_data_V6V8 = otu_data_V6V8[-777, ]

#link the sample ID to the metadata
##prepare the sample id for date comparison
sample_ids = rownames(otu_data_V6V8)
strsplit(gsub("BB", "", sample_ids), "\\.")

year = as.numeric(sapply(strsplit(gsub("BB", "", sample_ids), "\\."), function(x) x[[2]])) + 2000
week = as.numeric(sapply(strsplit(sample_ids, "\\."), function(x) substring(x[[3]], 1, nchar(x[[3]])-1)))
depth = sapply(sample_ids, function(x) substring(x, nchar(x), nchar(x)))
numeric_depth = c()
for (d in depth){
  if (d == "A"){
    numeric_depth = c(numeric_depth, 1)
  }else if (d == "B"){
    numeric_depth = c(numeric_depth, 5)
  }else if (d == "C"){
    numeric_depth = c(numeric_depth, 10)
  }else if (d == "D"){
    numeric_depth = c(numeric_depth, 60)
  }else if (d == "b"){
    numeric_depth = c(numeric_depth, 5)
  }else{
    print(d)
  }
}
basic_metadata_V6V8 = data.frame(sample_id = sample_ids, week = week, year = year, depth = numeric_depth)

##
date_exchange$...3 = sapply(as.POSIXct(as.numeric(date_exchange$...3) * (60*60*24), origin="1899-12-30"), function(x) strsplit(as.character(x), " ")[[1]][1])
basic_metadata_V6V8[["full_date"]] =sapply(1:nrow(basic_metadata_V6V8), 
                                           function(x) date_exchange$...3[which(date_exchange$METADATA == basic_metadata_V6V8[x, "week"] & 
                                                                                  date_exchange$...2 == basic_metadata_V6V8[x, "year"])])

##remove all the ids with "NA" in the week as this removes the duplicates
basic_metadata_V6V8 = basic_metadata_V6V8[rowSums(is.na(basic_metadata_V6V8)) == 0, ]
rm(depth, numeric_depth, d, week, year, sample_ids)

##prepare the metadata table for date comparisons
metadata = read.csv("./raw/bedford_basin_V4V5/CSV/bbmp_aggregated_profiles.csv")
metadata$date = sapply(strsplit(metadata$time_string, " "), function(x) x[[1]])

#where is the depth in the metadata??
counter = 1
for (i in 1:nrow(basic_metadata_V6V8)){
  this_date = basic_metadata_V6V8$full_date[i]
  this_depth = basic_metadata_V6V8$depth[i]
  this_index = basic_metadata_V6V8$sample_id[i]
  metadata_location = which(metadata$date == this_date & metadata$pressure == this_depth)
  
  if (length(metadata_location) != 0){
    this = metadata[metadata_location, ]
    this["id"] = this_index
    if(counter == 1){
      completed_metadata = this
    }else{
      completed_metadata = rbind(completed_metadata, this)
    }
    counter = counter + 1
  }
}

##you need to check the missing ones, here, again

rm(counter, this_depth, this_date, this_index, this, i, metadata_location)###rm a lot of things
rm(basic_metadata_V6V8, date_exchange, metadata)


###remove unnecessary metadata columns
rownames(completed_metadata) = completed_metadata$id
keep_columns = c("year_time", "month_time", "day_time", "pressure", "temperature", "conductivity", 
                 "fluorometer", "par", "sigmaTheta", "oxygen")
completed_metadata = completed_metadata[keep_columns]
completed_metadata = completed_metadata[rowSums(is.na(completed_metadata)) == 0, ]


##get rid of all the lines in the sequencing data I do not have metadata for
otu_data_V6V8 = otu_data_V6V8[rownames(otu_data_V6V8) %in% rownames(completed_metadata), ]

for (cc in colnames(otu_data_V6V8)){
  otu_data_V6V8[[cc]] = as.numeric(otu_data_V6V8[[cc]])
}
rm(cc)

remove_sample_ids = c("16SV6.BB16.42D")
otu_data_V6V8 = otu_data_V6V8[!(rownames(otu_data_V6V8) %in% remove_sample_ids), ]
completed_metadata = completed_metadata[rownames(otu_data_V6V8), ]

##create some plots and save the stuff
plot_bc_nmds_plots(otu_data_V6V8, completed_metadata, 5, selection_target_folder, "bedford_basin/V6V8")
write.csv(completed_metadata, paste(clean_target_folder, "bedford_basin/V6V8/metadata.csv", sep = "/"))
write.csv(otu_data_V6V8, paste(clean_target_folder, "bedford_basin/V6V8/otus.csv", sep = "/"))
write.csv(taxonomy_V6V8, paste(clean_target_folder, "bedford_basin/V6V8/taxonomy.csv", sep = "/"))
write.csv(sequences_V6V8, paste(clean_target_folder, "bedford_basin/V6V8/sequences.csv", sep = "/"),
          row.names = F, quote=FALSE)




###############
##boso_fish

setwd("/home/sperlea/Desktop/project_benchmark/data/new_datasets/")

otu_data = read.csv("./boso_fish/raw/asv_table.csv", row.names = 1)
metadata = read.csv("./boso_fish/raw/sample_sheet.csv", row.names = 1)

### remove NA samples 
metadata = metadata[!(is.na(metadata$water_temp) | is.na(metadata$salinity)), ]
otu_data = otu_data[row.names(metadata), ]
metadata = metadata[, c("water_temp", "salinity", "site_name")]

##make some PCoAs etc
# selection_target_folder = "/home/sperlea/Desktop/project_benchmark/data/new_datasets/boso_fish/figures"
# plot_bc_nmds_plots(otu_data, metadata, 4, selection_target_folder, "ff_")


#save -- plus: test/train split
train_ids = sample(row.names(otu_data), length(row.names(otu_data))*0.8)
test_ids = row.names(otu_data)[!(row.names(otu_data) %in% train_ids)]
  
train_otu_data = otu_data[train_ids, ]
test_otu_data = otu_data[test_ids, ]
train_meta_data = metadata[train_ids, ]
test_meta_data = metadata[test_ids, ]

write.csv(otu_data, "/home/sperlea/Desktop/project_benchmark/data/new_datasets/boso_fish/otu_table.csv")
write.csv(metadata, "/home/sperlea/Desktop/project_benchmark/data/new_datasets/boso_fish/metadata_table.csv")

write.csv(train_otu_data, "/home/sperlea/Desktop/project_benchmark/data/new_datasets/boso_fish/boso_fish/train/otus_train.csv")
write.csv(train_meta_data, "/home/sperlea/Desktop/project_benchmark/data/new_datasets/boso_fish/boso_fish/train/metadata_train.csv")

write.csv(test_otu_data, "/home/sperlea/Desktop/project_benchmark/data/new_datasets/boso_fish/boso_fish/test/otus_test.csv")
write.csv(test_meta_data, "/home/sperlea/Desktop/project_benchmark/data/new_datasets/boso_fish/boso_fish/test/metadata_test.csv")

