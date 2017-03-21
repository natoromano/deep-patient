SELECT n.patient_id, n.note_id, n.age_at_note_date_in_days as age, n.note_year, t.terms 
FROM stride6.notes n
JOIN
(SELECT nid, group_concat(case when negated=0 then tid else concat('N', tid) end separator " ") as terms
    FROM stride6.term_mentions
    WHERE familyHistory = 0
    GROUP BY nid) as t
ON n.note_id = t.nid;
