from os import name


def main():
    from databricks.connect import DatabricksSession
    spark = DatabricksSession.builder.getOrCreate()
    df = spark.createDataFrame(
        [(Pepito,10,)],
        ["nombre", "edad"]
    )
    df.write.saveAsTable("workspace,default,pepito")

    df.show()

if __name__ == "__main__":
    pass
else:
    main()