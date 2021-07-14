SELECT
  pub.publication_number as Publication_Number,
  ab.text as Abstract,
  STRING_AGG(citations.publication_number) as Citations
FROM
  patents-public-data.patents.publications as pub,
  UNNEST (abstract_localized) AS ab,
  UNNEST (citation) as citations
WHERE
  country_code = 'US'
  AND application_kind = 'A' -- A = patent
  And citations.type in ("A","D","E")
GROUP BY pub.publication_number, ab.text
LIMIT
  100