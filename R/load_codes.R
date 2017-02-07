###############################################################################
# (Script)
# Loads dumps from the STRIDE database and processes them according to the
# rules stated in the deep patient paper.
# cf README for more info.
#
# Nathanael Romano
###############################################################################

rm(list=ls())
library("dplyr")

# Load data
demo = as.data.frame(
  readRDS('/scratch/users/kjung/ehr-repr-learn/data/demographics.rds')
)
codes = as.data.frame(
  readRDS('/scratch/users/kjung/ehr-repr-learn/data/med.codes.rds')
)
orders = as.data.frame(
  readRDS('/scratch/users/kjung/ehr-repr-learn/data/rx.orders.rds')
)

# Fetch descriptors
all.codes = unique(codes$code)
all.orders = unique(orders$order)

# Filter years according to cutting point.
codes.2014 = filter(codes, visit_year == 2014)
codes = filter(codes, visit_year < 2014)
orders.2014 = filter(orders, start_time_year == 2014)
orders = filter(orders, start_time_year < 2014)

# Count codes before and after splitting point.
by_code <- group_by(codes.2014, patient_id, code)
codes.2014 <- as.data.frame(summarize(
  by_code, 
  count=n(), 
  age=mean(age_at_visit_in_days, na.rm=T)
))
by_code <- group_by(codes, patient_id, code)
codes <- as.data.frame(summarize(
  by_code, 
  count=n(), 
  age=mean(age_at_visit_in_days, na.rm=T)
))
by_code <- group_by(orders.2014, patient_id, rxcui)
orders.2014 <- as.data.frame(summarize(
  by_code, 
  count=n(), 
  age=mean(age_at_start_time_in_days, na.rm=T)
))
by_code <- group_by(orders, patient_id, rxcui)
orders <- as.data.frame(summarize(
  by_code, 
  count=n(), 
  age=mean(age_at_start_time_in_days, na.rm=T)
))

# Remove patients with too few records before 2014
ids <- union(orders$patient_id, codes$patient_id)
totalCounts <- data.frame(count=rep(0, length(ids)))
rownames(totalCounts) <- ids

by_patient <- group_by(orders, patient_id)
counts <- as.data.frame(summarize(by_patient, count=n()))
totalCounts[as.character(counts$patient_id), "count"] = 
  totalCounts[as.character(counts$patient_id), "count"] + counts[, "count"]

by_patient <- group_by(codes, patient_id)
counts <- as.data.frame(summarize(by_patient, count=n()))
totalCounts[as.character(counts$patient_id), "count"] = 
  totalCounts[as.character(counts$patient_id), "count"] + counts[, "count"]

# Patient IDs to keep
ids <- ids[totalCounts$count >= 5]
codes.2014 <- filter(codes.2014, patient_id %in% ids, !is.null(code))

# Remove codes appearing in less than 5 patients or more than 80% of patients.
by_code <- group_by(codes, code)
counts <- as.data.frame(summarize(by_code, count=n()))
codes.to.keep <- filter(counts, count >= 5, 
                        count < 0.8 * length(unique(codes$patient_id)))$code

by_patient <- group_by(orders, rxcui)
counts <- as.data.frame(summarize(by_code, count=n()))
orders.to.keep <- filter(counts, count >= 5, 
                         count < 0.8 * length(unique(orders$patient_id)))$rxcui


# Sample test patients, and hold them back from training set
test.patients = base::sample(unique(codes.2014$patient_id), size=80000)

codes = filter(codes, patient_id %in% ids, !is.null(code),
               code %in% codes.to.keep,
               !(patient_id %in% test.patients))
orders = filter(orders, patient_id %in% ids,!is.null(rxcui),
                rxcui %in% orders.to.keep,
                !(patient_id %in% test.patients))
demo = filter(demo, patient_id %in% ids, 
              !(patient_id %in% test.patients))

# Choose training patients for downstream task
train.patients = base::sample(unique(codes$patient_id), size=200000)
train.demo = filter(demo, patient_id %in% train.patients)
train.codes = filter(codes, patient_id %in% train.patients)
train.orders = filter(orders, patient_id %in% train.patients)
train.targets = filter(codes.2014, patient_id %in% train.patients)
# Add test patients for downstream task
test.demo = filter(demo, patient_id %in% test.patients)
test.codes = filter(codes, patient_id %in% test.patients)
test.orders = filter(orders, patient_id %in% test.patients)
test.targets = filter(codes.2014, patient_id %in% test.patients,
                      code %in% codes.to.keep)

# Save data
# Training data
write.table(codes,
          file='/scratch/users/naromano/deep-patient/shah/codes.csv',
          row.names=F)
write.table(orders, 
          file='/scratch/users/naromano/deep-patient/shah/orders.csv',
          row.name=F)
write.table(demo, 
          file='/scratch/users/naromano/deep-patient/shah/demo.csv',
          row.name=F)
# Disease prediction traininig data
write.table(train.demo,
          file='/scratch/users/naromano/deep-patient/shah/train.demo.csv',
          row.names=F)
write.table(train.codes,
          file='/scratch/users/naromano/deep-patient/shah/train.codes.csv',
          row.names=F)
write.table(train.orders,
          file='/scratch/users/naromano/deep-patient/shah/train.orders.csv',
          row.names=F)
write.table(train.targets,
          file='/scratch/users/naromano/deep-patient/shah/train.targets.csv',
          row.names=F)
# Disease prediction validation and test data
write.table(test.demo,
          file='/scratch/users/naromano/deep-patient/shah/test.demo.csv',
          row.names=F)
write.table(test.codes,
          file='/scratch/users/naromano/deep-patient/shah/test.codes.csv',
          row.names=F)
write.table(test.orders,
          file='/scratch/users/naromano/deep-patient/shah/test.orders.csv',
          row.names=F)
write.table(test.targets,
          file='/scratch/users/naromano/deep-patient/shah/test.targets.csv',
          row.names=F)

# Metadata
write(all.codes, file='../data/codes.txt', sep='\n')
write(all.orders, file='../data/orders.txt', sep=', ')
write(ids, file='../data/patients.txt', sep='\n')
write(train.patients, file='../data/train.patients.txt', sep='\n')
write(test.patients, file='../data/test.patients.txt', sep='\n')
