import * as cdk from 'aws-cdk-lib';
import * as cognito from 'aws-cdk-lib/aws-cognito';
import * as appsync from 'aws-cdk-lib/aws-appsync';
import * as dynamodb from 'aws-cdk-lib/aws-dynamodb';
import { Construct } from 'constructs';

export class NetflixCloneStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props?: cdk.StackProps) {
    super(scope, id, props);

    // Create the User Pool
    const userPool = new cognito.UserPool(this, 'NetflixCloneUserPool', {
      userPoolName: 'netflix-clone-user-pool',
      selfSignUpEnabled: true,
      autoVerify: {
        email: false,
      },
      standardAttributes: {
        email: {
          required: true,
          mutable: true,
        },
      },
      passwordPolicy: {
        minLength: 8,
        requireLowercase: true,
        requireUppercase: true,
        requireDigits: true,
        requireSymbols: true,
      },
      accountRecovery: cognito.AccountRecovery.EMAIL_ONLY,
      removalPolicy: cdk.RemovalPolicy.DESTROY,
    });

    // Create the App Client
    const userPoolClient = new cognito.UserPoolClient(this, 'NetflixCloneUserPoolClient', {
      userPool,
      authFlows: {
        userPassword: true,
        userSrp: true,
      },
      oAuth: {
        flows: {
          authorizationCodeGrant: true,
        },
        scopes: [
          cognito.OAuthScope.EMAIL,
          cognito.OAuthScope.OPENID,
          cognito.OAuthScope.PROFILE,
        ],
        callbackUrls: ['http://localhost:3000/'],
        logoutUrls: ['http://localhost:3000/login'],
      },
    });

    // Create the User Pool Domain
    const domain = userPool.addDomain('NetflixCloneDomain', {
      cognitoDomain: {
        domainPrefix: 'netflix-clone',
      },
    });

    // Reference existing DynamoDB tables
    const moviesTable = dynamodb.Table.fromTableName(
      this,
      'ExistingMoviesTable',
      'movie-recommender-movies'
    );

    const movieLinksTable = dynamodb.Table.fromTableName(
      this,
      'ExistingMovieLinksTable',
      'movie-recommender-links'
    );

    const ratingsTable = dynamodb.Table.fromTableName(
      this,
      'ExistingRatingsTable',
      'movie-recommender-ratings'
    );

    // Create the AppSync API
    const api = new appsync.CfnGraphQLApi(this, 'NetflixCloneAPI', {
      name: 'netflix-clone-api',
      authenticationType: 'AMAZON_COGNITO_USER_POOLS',
      userPoolConfig: {
        userPoolId: userPool.userPoolId,
        awsRegion: this.region,
        defaultAction: 'ALLOW'
      }
    });

    // Create the Schema
    const apiSchema = new appsync.CfnGraphQLSchema(this, 'NetflixCloneSchema', {
      apiId: api.attrApiId,
      definition: `
        type Movie {
          movieId: Int!
          title: String!
          genres: String!
        }

        type Rating {
          userId: String!
          movieId: Int!
          rating: Float!
          timestamp: Int!
        }

        type Query {
          listMovies: [Movie!]!
          listRatings: [Rating!]!
          listMovieLinks: [Movie!]!
        }

        input CreateRatingInput {
          userId: String!
          movieId: Int!
          rating: Float!
          timestamp: Int!
        }

        type Mutation {
          createRating(input: CreateRatingInput!): Rating
        }
      `
    });

    // Create DynamoDB data sources
    const moviesDS = new appsync.CfnDataSource(this, 'MoviesDataSource', {
      apiId: api.attrApiId,
      name: 'MoviesTable',
      type: 'AMAZON_DYNAMODB',
      dynamoDbConfig: {
        tableName: moviesTable.tableName,
        awsRegion: this.region
      },
      serviceRoleArn: new cdk.aws_iam.Role(this, 'MoviesDSRole', {
        assumedBy: new cdk.aws_iam.ServicePrincipal('appsync.amazonaws.com'),
        inlinePolicies: {
          'DynamoDBAccess': new cdk.aws_iam.PolicyDocument({
            statements: [
              new cdk.aws_iam.PolicyStatement({
                actions: ['dynamodb:Scan'],
                resources: [moviesTable.tableArn]
              })
            ]
          })
        }
      }).roleArn
    });

    const movieLinksDS = new appsync.CfnDataSource(this, 'MovieLinksDataSource', {
      apiId: api.attrApiId,
      name: 'MovieLinksTable',
      type: 'AMAZON_DYNAMODB',
      dynamoDbConfig: {
        tableName: movieLinksTable.tableName,
        awsRegion: this.region
      },
      serviceRoleArn: new cdk.aws_iam.Role(this, 'MovieLinksDSRole', {
        assumedBy: new cdk.aws_iam.ServicePrincipal('appsync.amazonaws.com'),
        inlinePolicies: {
          'DynamoDBAccess': new cdk.aws_iam.PolicyDocument({
            statements: [
              new cdk.aws_iam.PolicyStatement({
                actions: ['dynamodb:Scan'],
                resources: [movieLinksTable.tableArn]
              })
            ]
          })
        }
      }).roleArn
    });

    const ratingsDS = new appsync.CfnDataSource(this, 'RatingsDataSource', {
      apiId: api.attrApiId,
      name: 'RatingsTable',
      type: 'AMAZON_DYNAMODB',
      dynamoDbConfig: {
        tableName: ratingsTable.tableName,
        awsRegion: this.region
      },
      serviceRoleArn: new cdk.aws_iam.Role(this, 'RatingsDSRole', {
        assumedBy: new cdk.aws_iam.ServicePrincipal('appsync.amazonaws.com'),
        inlinePolicies: {
          'DynamoDBAccess': new cdk.aws_iam.PolicyDocument({
            statements: [
              new cdk.aws_iam.PolicyStatement({
                actions: ['dynamodb:Scan', 'dynamodb:PutItem'],
                resources: [ratingsTable.tableArn]
              })
            ]
          })
        }
      }).roleArn
    });

    // Add explicit dependencies
    const listMoviesResolver = new appsync.CfnResolver(this, 'ListMoviesResolver', {
      apiId: api.attrApiId,
      typeName: 'Query',
      fieldName: 'listMovies',
      dataSourceName: moviesDS.attrName,
      requestMappingTemplate: `{
        "version": "2018-05-29",
        "operation": "Scan"
      }`,
      responseMappingTemplate: `$util.toJson($ctx.result.items)`
    });
    listMoviesResolver.addDependsOn(moviesDS);

    const listMovieLinksResolver = new appsync.CfnResolver(this, 'ListMovieLinksResolver', {
      apiId: api.attrApiId,
      typeName: 'Query',
      fieldName: 'listMovieLinks',
      dataSourceName: movieLinksDS.attrName,
      requestMappingTemplate: `{
        "version": "2018-05-29",
        "operation": "Scan"
      }`,
      responseMappingTemplate: `$util.toJson($ctx.result.items)`
    });
    listMovieLinksResolver.addDependsOn(movieLinksDS);

    const listRatingsResolver = new appsync.CfnResolver(this, 'ListRatingsResolver', {
      apiId: api.attrApiId,
      typeName: 'Query',
      fieldName: 'listRatings',
      dataSourceName: ratingsDS.attrName,
      requestMappingTemplate: `{
        "version": "2018-05-29",
        "operation": "Scan"
      }`,
      responseMappingTemplate: `$util.toJson($ctx.result.items)`
    });
    listRatingsResolver.addDependsOn(ratingsDS);

    const createRatingResolver = new appsync.CfnResolver(this, 'CreateRatingResolver', {
      apiId: api.attrApiId,
      typeName: 'Mutation',
      fieldName: 'createRating',
      dataSourceName: ratingsDS.attrName,
      requestMappingTemplate: `{
        "version": "2018-05-29",
        "operation": "PutItem",
        "key": {
          "userId": $util.dynamodb.toDynamoDBJson($ctx.args.input.userId),
          "movieId": $util.dynamodb.toDynamoDBJson($ctx.args.input.movieId)
        },
        "attributeValues": $util.dynamodb.toMapValuesJson($ctx.args.input)
      }`,
      responseMappingTemplate: `$util.toJson($ctx.result)`
    });
    createRatingResolver.addDependsOn(ratingsDS);

    // Output the values needed for the frontend
    new cdk.CfnOutput(this, 'UserPoolId', {
      value: userPool.userPoolId,
    });

    new cdk.CfnOutput(this, 'UserPoolClientId', {
      value: userPoolClient.userPoolClientId,
    });

    new cdk.CfnOutput(this, 'CognitoDomain', {
      value: domain.domainName,
    });

    new cdk.CfnOutput(this, 'GraphQLApiEndpoint', {
      value: api.attrGraphQlUrl,
    });

    new cdk.CfnOutput(this, 'GraphQLApiId', {
      value: api.attrApiId,
    });
  }
} 