
POSTGRE_BLOCKING = """
SELECT datname
, usename
, wait_event_type
, wait_event
, pg_blocking_pids(pid) AS blocked_by
, query
FROM pg_stat_activity
WHERE wait_event IS NOT NULL;
"""

POSTGRE_TRANSACTIONS = """
SELECT
  pid,
  now() - pg_stat_activity.query_start AS duration,
  query,
  state
FROM pg_stat_activity
WHERE (now() - pg_stat_activity.query_start) > interval '5 minutes';
"""

# Replace   {<pid>}   with your pid
POSTGRE_CANCEL = """
SELECT pg_cancel_backend({<pid>});
"""
POSTGRE_TERMINATE = """
SELECT pg_terminate_backend({<pid>});
"""

# Clean up of old entries in the table
POSTGRE_ = """
DELETE FROM {<table>}
WHERE {<entry_timestamp>} < now() - interval '30 days'
"""
