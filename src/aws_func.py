#!/usr/bin/env python3

import os
import sys
import argparse
import configparser

CREDENTIALS_PATH = "~/.aws/credentials"


def fail(message):
    sys.stderr.write(message + "\n")
    sys.stderr.flush()
    sys.exit(1)


def export_aws_profiles():
    """
    Export AWS CLI Profiles to environment Variables
    """
    parser = argparse.ArgumentParser(
        prog="aws-env",
        description="Extract AWS credentials from a given profile to the environment variables.")
    parser.add_argument(
        '-n', '--no-export', action="store_true",
        help="Do not use export on the variables.")
    parser.add_argument(
        "prod",
        help="The profile in ~/.aws/credentials to extract credentials for.")

    args = parser.parse_args()

    # Load ini file into dictionary
    config = configparser.ConfigParser()

    if not os.path.isfile(os.path.expanduser(CREDENTIALS_PATH)):
        fail("Unable to load credentials from file: {}".format(os.path.expanduser(CREDENTIALS_PATH)))

    config.read(os.path.expanduser(CREDENTIALS_PATH))

    # sanity checking
    if args.profile not in config.sections():
        fail("Profile '{}' does not exist.".format(args.profile))

    if 'aws_access_key_id' not in config[args.profile].keys():
        fail("AWS Access Key ID not found in profile '{}'".format(args.profile))

    if 'aws_secret_access_key' not in config[args.profile].keys():
        fail("AWS Secret Access Key not found in profile '{}'".format(args.profile))

    if args.no_export:
        export=""
    else:
        export="export "

    sys.stdout.write("{}AWS_ACCESS_KEY_ID={}\n".format(export, config[args.profile]['aws_access_key_id']))
    sys.stdout.write("{}AWS_SECRET_ACCESS_KEY={}\n".format(export, config[args.profile]['aws_secret_access_key']))
    sys.stdout.flush()
