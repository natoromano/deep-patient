###############################################################################
# Loads dumps from the STRIDE database and processes them to construct 
# matrices of code counts
#
# Authors : Nathanael Romano, Sebastien Dubois
###############################################################################

rm(list=ls())
library("dplyr")


###############################################################################
# LOAD DATA
###############################################################################

demo = as.data.frame(
  readRDS('/scratch/users/kjung/ehr-repr-learn/data/demographics.rds')
)
meds = as.data.frame(
  readRDS('/scratch/users/kjung/ehr-repr-learn/data/billing.codes.input.rds')
)
orders = as.data.frame(
  readRDS('/scratch/users/kjung/ehr-repr-learn/data/rx.orders.input.rds')
)
codes.targets = as.data.frame(
  readRDS('/scratch/users/kjung/ehr-repr-learn/data/billing.codes.target.rds')
)

# Fetch descriptors
all.meds = unique(meds$code)
all.orders = unique(orders$rxcui)


###############################################################################
# AGGREGATE CODES
###############################################################################

# Count codes before and after splitting point
codes.targets <- codes.targets %>%
  filter(sab %in% c("BILLING", "DX_ID")) %>%
  group_by(patient_id, code) %>%
  summarize(
    count = n(),
    age = mean(age_at_visit_in_days, na.rm=T)
  )

meds <- meds %>%
  group_by(patient_id, code) %>%
  summarize(
    count = n(),
    age = mean(age_at_visit_in_days, na.rm=T)
  )

orders <- orders %>%
  group_by(patient_id, rxcui) %>%
  summarize(
    count = n(),
    age = mean(age_at_start_time_in_days, na.rm=T)
  )


###############################################################################
# FILTER PATIENTS AND CODES
###############################################################################

# Select codes 
MAX_PATIENTS_M = 0.8 * n_distinct(meds$patient_id)
MIN_PATIENTS_M = 25  # 0.001 * n_distinct(meds$patient_id)
meds.to.keep <- meds %>% 
  group_by(code) %>% 
  summarize(count = n()) %>% 
  filter(count >= MIN_PATIENTS_M, count < MAX_PATIENTS_M) %>% 
  .$code

MAX_PATIENTS_O = 0.8 * n_distinct(orders$patient_id)
MIN_PATIENTS_O = 5  # 0.001 * n_distinct(orders$patient_id)
orders.to.keep <- orders %>% 
  group_by(rxcui) %>% 
  summarize(count = n()) %>% 
  filter(count >= MIN_PATIENTS_O, count < MAX_PATIENTS_O) %>% 
  .$rxcui

orders <- orders %>% 
  filter(rxcui %in% orders.to.keep)
meds <- meds %>% 
  filter(code %in% meds.to.keep)

# Patient IDs to keep, only look at med codes
MIN_COUNT = 3
ids <- meds %>% 
  group_by(patient_id) %>% 
  summarize(count = n()) %>% 
  filter(count >= MIN_COUNT) %>% 
  .$patient_id

print(length(ids))


###############################################################################
# ASSEMBLE INPUT DATA
###############################################################################

# Transform orders and meds into matrices where each column corresponds
# to a code
demo <- demo %>%
  filter(patient_id %in% ids) %>%
  mutate("gender" = (gender == "MALE") * 1.0) %>%
  select(patient_id, gender, age_at_death_in_days)

med_matrix <- meds %>% 
  select(patient_id, code, count) %>%
  mutate("code" = paste("MED", code, sep="")) %>%
  reshape2::dcast(patient_id ~ code, fill = 0, value.var = "count") %>% 
  filter(patient_id %in% ids)

rx_matrix <- orders %>% 
  select(patient_id, rxcui, count) %>%
  mutate("rxcui" = paste("RX", rxcui, sep="")) %>%
  reshape2::dcast(patient_id ~ rxcui, fill = 0, value.var = "count") %>% 
  filter(patient_id %in% ids)

# Extract some metadata
rxcodes <- colnames(rx_matrix)
rxcodes <- rxcodes[order(rxcodes)]

medcodes <- colnames(med_matrix)
medcodes <- medcodes[order(medcodes)]
  
# Extract target labels
codes.targets <- codes.targets %>% 
  select(patient_id, code, count)

# Assemble matrix
x_matrix <- demo
x_matrix <- x_matrix %>%
  merge(med_matrix, by="patient_id", all.x=T) %>%
  merge(rx_matrix, by="patient_id", all.x=T, sort=T)
x_matrix[is.na(x_matrix)] <- 0
x_matrix <- x_matrix[, order(colnames(x_matrix))]


###############################################################################
# EXTRACT TARGET TASK LABELS
###############################################################################

# Replace with CCS mappings
ccs <- read.table(
  "/scratch/users/kjung/ehr-repr-learn/data/baseline_ccs2icd9.txt"
)
codes.targets$mergedCode <- gsub("\\.", "", codes.targets$code)
codes.targets <- codes.targets %>%
  merge(ccs, by.x="mergedCode", by.y="ICD9.CODE", all=F) %>%
  select(patient_id, CCS.CODE, count) %>%
  reshape2::dcast(formula=patient_id ~ CCS.CODE, fill=0, value.var="count",
                  fun.aggregate=sum)
y_matrix <- merge(demo, codes.targets, by="patient_id", all.x=T)
y_matrix[y_matrix > 1] = 1
y_matrix[is.na(y_matrix)] <- 0


###############################################################################
# DATASET SPLIT
###############################################################################

print(nrow(x_matrix))

test_size <- 50000
val_size <- 10000
train_size <- length(ids) - val_size - test_size

# Sample  patients
set.seed(0)
all.patients <- base::sample(ids, replace=F)
test.patients <- all.patients[1:test_size]
val.patients <- all.patients[test_size:(test_size + val_size)]
train.patients <- all.patients[(test_size + val_size):(length(ids))]

# Create split data frame
split.ids <- data.frame(patient_id=ids[order(ids)], train=0, val=0, test=0)
split.ids[split.ids$patient_id %in% train.patients, "train"] <- 1
split.ids[split.ids$patient_id %in% val.patients, "val"] <- 1
split.ids[split.ids$patient_id %in% test.patients, "test"] <- 1

# Extract code names in order for later compatibility
all.codes <- colnames(x_matrix)
all.labels <- colnames(y_matrix)

# Train
x.train <- filter(x_matrix, patient_id %in% train.patients)
y.train <- filter(y_matrix, patient_id %in% train.patients)

# Val
x.val <- filter(x_matrix, patient_id %in% val.patients)
y.val <- filter(y_matrix, patient_id %in% val.patients)

# Test
x.test <- filter(x_matrix, patient_id %in% test.patients)
y.test <- filter(y_matrix, patient_id %in% test.patients)


###############################################################################
# SAVE DATA
###############################################################################

# Save data
write.table(x.train,
            file='/scratch/users/kjung/ehr-repr-learn/data/x.train.txt',
            row.names=F)
write.table(y.train,
            file='/scratch/users/kjung/ehr-repr-learn/data/y.train.txt',
            row.names=F)
write.table(x.val,
            file='/scratch/users/kjung/ehr-repr-learn/data/x.val.txt',
            row.names=F)
write.table(y.val,
            file='/scratch/users/kjung/ehr-repr-learn/data/y.val.txt',
            row.names=F)
write.table(x.test,
            file='/scratch/users/kjung/ehr-repr-learn/data/x.test.txt',
            row.names=F)
write.table(y.test,
            file='/scratch/users/kjung/ehr-repr-learn/data/y.test.txt',
            row.names=F)

# Save metadata
write.table(split.ids, 
            file='/scratch/users/kjung/ehr-repr-learn/data/split.txt',
            row.names=F)
write(ids, 
      file='/scratch/users/kjung/ehr-repr-learn/data/patients.txt', 
      sep='\n')
write(all.codes,
      file='/scratch/users/kjung/ehr-repr-learn/data/x.covariates.txt',
      sep='\n')
write(all.labels,
      file='/scratch/users/kjung/ehr-repr-learn/data/y.labels.txt',
      sep='\n')
write(medcodes,
      file='/scratch/users/kjung/ehr-repr-learn/data/medcodes.txt',
      sep='\n')
write(rxcodes,
      file='/scratch/users/kjung/ehr-repr-learn/data/rxcuis.txt',
      sep='\n')
