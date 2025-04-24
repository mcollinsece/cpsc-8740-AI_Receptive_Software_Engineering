#!/usr/bin/env node
import 'source-map-support/register';
import * as cdk from 'aws-cdk-lib';
import { NetflixCloneStack } from './lib/netflix-clone-stack';

const app = new cdk.App();
new NetflixCloneStack(app, 'NetflixCloneStack', {
  env: {
    account: process.env.CDK_DEFAULT_ACCOUNT,
    region: process.env.CDK_DEFAULT_REGION || 'us-east-1',
  },
}); 