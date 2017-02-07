SELECT nid, group_concat(distinct tid separator " ") as terms
FROM stride6.term_mentions
WHERE negated = 0 and familyHistory = 0
GROUP BY nid;
