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

# Path so save the data and metadata
REPO_PATH = '/home/naromano/deep-patient'
PATH = '/scratch/users/naromano/deep-patient/shah'
METADATA_PATH = paste(REPO_PATH, 'data', sep='/')

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

by_code <- group_by(orders, rxcui)
counts <- as.data.frame(summarize(by_code, count=n()))
orders.to.keep <- filter(counts, count >= 5, 
                         count < 0.8 * length(unique(orders$patient_id)))$rxcui


# Sample test patients, and hold them back from training set
test.patients = unique(codes.2014$patient_id)
test.patients = test.patients[test.patients %in% 
                                  rownames(filter(totalCounts, count >= 10))]
test.patients = base::sample(unique(codes.2014$patient_id), size=80000)

# Add test patients for downstream task
test.demo = filter(demo, patient_id %in% test.patients)
test.codes = filter(codes, patient_id %in% test.patients)
test.orders = filter(orders, patient_id %in% test.patients)
test.targets = filter(codes.2014, patient_id %in% test.patients,
                      code %in% codes.to.keep)

# Hold back test patients from source task training set
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

# Save data
# Training data
write.table(codes,
            file=paste(PATH, 'codes.csv', sep='/'),
            row.names=F)
write.table(orders, 
            file=paste(PATH, 'orders.csv', sep='/'),
            row.name=F)
write.table(demo, 
            file=paste(PATH, 'demo.csv', sep='/'),
            row.name=F)
# Disease prediction traininig data
write.table(train.demo,
            file=paste(PATH, 'train.demo.csv', sep='/'),
            row.names=F)
write.table(train.codes,
            file=paste(PATH, 'train.codes.csv', sep='/'),
            row.names=F)
write.table(train.orders,
            file=paste(PATH, 'train.orders.csv', sep='/'),
            row.names=F)
write.table(train.targets,
            file=paste(PATH, 'train.targets.csv', sep='/'),
            row.names=F)
# Disease prediction validation and test data
write.table(test.demo,
            file=paste(PATH, 'test.demo.csv', sep='/'),
            row.names=F)
write.table(test.codes,
            file=paste(PATH, 'test.codes.csv', sep='/'),
            row.names=F)
write.table(test.orders,
            file=paste(PATH, 'test.orders.csv', sep='/'),
            row.names=F)
write.table(test.targets,
            file=paste(PATH, 'test.targets.csv', sep='/'),
            row.names=F)

# Metadata
write(all.codes, file=paste(METADATA_PATH, 'codes.txt', sep='/'), sep='\n')
write(all.orders, file=paste(METADATA_PATH, 'orders.txt', sep='/'), sep=', ')
write(ids, file=paste(METADATA_PATH, 'patients.txt', sep='/'), sep='\n')
write(train.patients,
      file=paste(METADATA_PATH, 'train.patients.txt', sep='/'), sep='\n')
write(test.patients,
      file=paste(METADATA_PATH, 'test.patients.txt', sep='/'), sep='\n')
